#!/usr/bin/env bash
# Extract the same utterances with the fine-tuned encoder, for a controlled
# comparison against the frozen one.
#
# The Kaggle figures (macro F1 0.9186) come from that notebook's own held-out
# split of a deliberately class-balanced pack, scored by a linear head trained
# jointly with the encoder. Ours (0.863) come from a HistGradientBoosting
# classifier on naturally distributed features from whole utterances. Four
# things differ, so neither number says anything about the other.
#
# This holds everything constant except the encoder: identical utterances,
# identical slicing, identical downstream classifier and speaker split. Only
# then does a difference mean the fine-tuning worked.
#
#   bash scripts/extract_finetuned_subset.sh          # first 3 slices
#   SLICES="0 4000 8000 12000" bash scripts/extract_finetuned_subset.sh
set -u

source "$(dirname "${BASH_SOURCE[0]}")/_repo.sh"
# Slice caches are written to paths relative to the repository, so run from
# there rather than from wherever this was invoked.
cd "$ROOT"
OUT=${OUT:-models/w2v_slices_v3_ft}
MODEL=${MODEL:-models/finetuned_encoder}
SLICES=${SLICES:-"0 4000 8000"}
BUDGET=${BUDGET:-3000}

mkdir -p "$OUT"
for offset in $SLICES; do
  target="$OUT/slice_$(printf '%06d' "$offset").npz"
  if [[ -s "$target" ]]; then
    echo "$(date +%H:%M:%S) | slice $offset already done"
    continue
  fi
  echo "$(date +%H:%M:%S) | fine-tuned slice $offset"
  timeout --signal=KILL "$BUDGET" "$PY" -u scripts/extract_wav2vec_features.py \
    --model "$MODEL" \
    --corpora "output/aligned_v3:data/train_corpus" \
    --layers 16 --device cuda \
    --offset "$offset" --take 4000 --out "$target"
  echo "$(date +%H:%M:%S) | slice $offset exit $?"
done
echo "$(date +%H:%M:%S) | complete: $(ls "$OUT" | wc -l) slices"
