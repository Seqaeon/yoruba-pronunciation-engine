"""Tests for Goodness-of-Pronunciation scoring."""

from __future__ import annotations

import csv
import sqlite3

import pytest

from yoruba_engine.alignment.gop import (
    FRAME_SHIFT_S,
    GOPReference,
    PhoneScore,
    read_scores,
)


class TestPhoneScore:
    def test_per_frame_normalises_by_duration(self):
        """A long and a short phone with the same total must not score alike.

        MFA reports a total log-likelihood over the span, so raw values scale
        with duration -- measured, a 0.82 s span scored -58 while a 0.04 s span
        scored -83. Without this the short phone looks better.
        """
        long_phone = PhoneScore("a", 0.0, 0.80, -80.0)     # 80 frames
        short_phone = PhoneScore("a", 0.0, 0.10, -80.0)    # 10 frames
        assert long_phone.per_frame == pytest.approx(-1.0)
        assert short_phone.per_frame == pytest.approx(-8.0)
        assert long_phone.per_frame > short_phone.per_frame

    def test_frames_never_zero(self):
        """A span shorter than one frame must still divide by at least 1."""
        assert PhoneScore("k", 0.0, 0.001, -5.0).frames == 1

    def test_frames_matches_frame_shift(self):
        assert PhoneScore("a", 0.0, 10 * FRAME_SHIFT_S, -1.0).frames == 10


class TestGOPReference:
    def test_z_score_arithmetic(self):
        reference = GOPReference({"a": (-5.0, 2.0)})
        # per_frame of -7.0 is one standard deviation below the mean.
        assert reference.z_score(PhoneScore("a", 0.0, 0.10, -70.0)) == pytest.approx(-1.0)

    def test_unknown_phone_yields_no_opinion(self):
        """A phone with no statistics must return None, not a pass or a fail.

        Treating "not measured" as "fine" would silently exempt whole phones
        from the check.
        """
        reference = GOPReference({"a": (-5.0, 2.0)})
        assert reference.z_score(PhoneScore("gb", 0.0, 0.1, -70.0)) is None

    def test_zero_deviation_yields_no_opinion(self):
        """A degenerate distribution cannot produce a z-score."""
        reference = GOPReference({"a": (-5.0, 0.0)})
        assert reference.z_score(PhoneScore("a", 0.0, 0.1, -70.0)) is None

    def test_empty_reference_is_silent(self):
        assert GOPReference().z_score(PhoneScore("a", 0.0, 0.1, -1.0)) is None

    def test_round_trips_through_disk(self, tmp_path):
        original = GOPReference({"a": (-5.652, 4.264), "un": (-8.498, 7.917)})
        path = original.save(tmp_path / "reference.json")
        loaded = GOPReference.load(path)
        assert loaded.statistics == original.statistics

    def test_missing_file_loads_empty_rather_than_raising(self, tmp_path):
        """A missing reference disables the check; it must not break assessment."""
        assert GOPReference.load(tmp_path / "absent.json").statistics == {}


class TestReadScores:
    def _write_run(self, root, rows, phones, utterances):
        alignment = root / "corpus" / "alignment"
        alignment.mkdir(parents=True)
        with (alignment / "phone_intervals.csv").open("w", newline="", encoding="utf-8") as h:
            writer = csv.DictWriter(h, fieldnames=[
                "id", "begin", "end", "phone_goodness", "phone_id", "utterance_id"])
            writer.writeheader()
            writer.writerows(rows)
        database = root / "corpus" / "corpus.db"
        connection = sqlite3.connect(database)
        connection.execute("CREATE TABLE phone (id INTEGER, phone TEXT)")
        connection.executemany("INSERT INTO phone VALUES (?, ?)", phones)
        connection.execute("CREATE TABLE file (id INTEGER, name TEXT)")
        connection.execute("CREATE TABLE utterance (id INTEGER, file_id INTEGER)")
        for utterance_id, name in utterances:
            connection.execute("INSERT INTO file VALUES (?, ?)", (utterance_id, name))
            connection.execute("INSERT INTO utterance VALUES (?, ?)",
                               (utterance_id, utterance_id))
        connection.commit()
        connection.close()

    def test_reads_and_maps_ids_to_labels(self, tmp_path):
        self._write_run(
            tmp_path,
            rows=[
                {"id": 1, "begin": 0.0, "end": 0.8, "phone_goodness": -58.0,
                 "phone_id": 2, "utterance_id": 1},
                {"id": 2, "begin": 0.8, "end": 0.9, "phone_goodness": -10.0,
                 "phone_id": 5, "utterance_id": 1},
            ],
            phones=[(2, "sil"), (5, "a")],
            utterances=[(1, "utt_one")],
        )
        scores = read_scores(tmp_path, "corpus")
        assert list(scores) == ["utt_one"]
        # The silence row is dropped: it carries no pronunciation judgement.
        assert [p.phone for p in scores["utt_one"]] == ["a"]

    def test_orders_phones_by_time(self, tmp_path):
        self._write_run(
            tmp_path,
            rows=[
                {"id": 1, "begin": 0.5, "end": 0.6, "phone_goodness": -5.0,
                 "phone_id": 5, "utterance_id": 1},
                {"id": 2, "begin": 0.1, "end": 0.2, "phone_goodness": -6.0,
                 "phone_id": 6, "utterance_id": 1},
            ],
            phones=[(5, "a"), (6, "k")],
            utterances=[(1, "utt_one")],
        )
        scores = read_scores(tmp_path, "corpus")
        assert [p.phone for p in scores["utt_one"]] == ["k", "a"]

    def test_missing_artifacts_return_empty(self, tmp_path):
        """A cleaned-up MFA run must degrade quietly, not raise."""
        assert read_scores(tmp_path, "corpus") == {}
