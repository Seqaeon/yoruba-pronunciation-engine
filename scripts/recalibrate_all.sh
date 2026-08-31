#!/usr/bin/env bash
# Recalibrate every threshold that reads classifier probabilities.
#
# GOP and min_duration_fraction are deliberately absent: GOP scores come from
# the acoustic model and duration fractions from alignment, neither of which the
# encoder swap touches. Re-running them would only add noise to numbers that are
# already correct.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_repo.sh"
cd "$ROOT"

echo "##### reliability: duration floor and entropy ceiling #####"
timeout 3600 $PY scripts/calibrate_reliability.py --aligned output/slr86_new --limit 400 2>&1 | tail -26
echo; echo "##### suspect and weak probability floors #####"
timeout 5400 $PY scripts/calibrate_suspect.py --limit 250 2>&1 | tail -22
echo; echo "##### decoder margin #####"
timeout 5400 $PY scripts/calibrate_decoder.py --limit 250 2>&1 | tail -22
echo; echo "##### done #####"
