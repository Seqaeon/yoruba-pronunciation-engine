#!/usr/bin/env bash
# Extract wav2vec features in slices, surviving GPU hangs.
#
# A single 5-hour extraction died after an hour: the process kept a full core
# busy with the GPU at 0% utilisation, zero read syscalls, and no output for two
# hours. Nothing in the data explained it -- the utterances around the stall were
# ordinary 9-second 16 kHz clips -- so it is a driver-level hang under sustained
# load, which laptop GPUs are prone to. Whatever the cause, an extraction that
# loses everything when it happens is the wrong shape for this machine.
#
# Each slice is a separate process writing its own cache file, wrapped in a
# timeout. A hang costs one slice, the timeout kills it, and the retry starts a
# fresh CUDA context. Slices already on disk are skipped, so the whole run is
# resumable.
#
#   bash scripts/extract_features_sliced.sh
set -u

PY=/home/seqaeon/Downloads/venv/bin/python
OUT=models/w2v_slices
TOTAL=55665
SLICE=${SLICE:-4000}
# A slice of 4,000 runs about 27 minutes at the observed 2.5 utt/s; 50 minutes
# leaves room for a slow one without waiting out a hang.
BUDGET=${BUDGET:-3000}
RETRIES=${RETRIES:-2}

CORPORA=(
  "output/aligned_v2/slr86:data/train_corpus"
  "output/aligned_v2/studio:data/train_corpus"
  "output/aligned_v2/yecs:data/train_corpus"
  "output/aligned_v2/mich:data/train_corpus"
)

mkdir -p "$OUT"
echo "$(date +%H:%M:%S) | extracting $TOTAL utterances in slices of $SLICE"

for (( offset=0; offset<TOTAL; offset+=SLICE )); do
  target="$OUT/slice_$(printf '%06d' "$offset").npz"
  if [[ -s "$target" ]]; then
    echo "$(date +%H:%M:%S) | slice $offset already done, skipping"
    continue
  fi
  for (( attempt=1; attempt<=RETRIES+1; attempt++ )); do
    echo "$(date +%H:%M:%S) | slice $offset attempt $attempt"
    timeout --signal=KILL "$BUDGET" "$PY" -u scripts/extract_wav2vec_features.py \
      --corpora "${CORPORA[@]}" \
      --layers 16 --device cuda \
      --offset "$offset" --take "$SLICE" \
      --out "$target"
    status=$?
    if [[ $status -eq 0 && -s "$target" ]]; then
      echo "$(date +%H:%M:%S) | slice $offset done"
      break
    fi
    # 137 is the KILL the timeout sends; anything else is a real failure, but
    # both are retried because a fresh process gets a fresh CUDA context.
    echo "$(date +%H:%M:%S) | slice $offset failed (status $status), retrying"
    rm -f "$target"
    sleep 10
  done
done

echo "$(date +%H:%M:%S) | slices complete: $(ls "$OUT" | wc -l)"
