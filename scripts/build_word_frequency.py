"""Count Yoruba word frequencies across every corpus on disk.

The decoder scores candidate words by acoustics alone, which lets it pick words
that do not occur in the language as used: ``ọ̀kà`` was decoded for a learner
who said ``ọkàn`` despite appearing zero times in 209,265 corpus tokens, while
``ọkàn`` appears 202 times. A unigram prior is the standard fix and costs one
number per word.
"""
import csv, json, sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import pyarrow.parquet as pq
from yoruba_engine.g2p.orthography import normalize, tokenize

counts = Counter()

def add(texts):
    for text in texts:
        counts.update(normalize(w) for w in tokenize(text or "", keep_noise=False))

data = PROJECT_ROOT / "data"
index = data / "corpus" / "line_index.tsv"
if index.exists():
    with index.open(encoding="utf-8", newline="") as handle:
        add([r[2] for r in csv.reader(handle, delimiter="\t") if len(r) >= 3])

iroyin = data / "IroyinSpeech" / "validated.tsv"
if iroyin.exists():
    with iroyin.open(encoding="utf-8", newline="") as handle:
        add([r.get("sentence") or "" for r in csv.DictReader(handle, delimiter="\t")])

for path in sorted(data.glob("*.parquet")):
    seen = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
        d = batch.to_pydict()
        add(d.get("sentence") or d.get("text") or [])
        seen += 512
        if seen > 60000:
            break

counts.pop("", None)
out = PROJECT_ROOT / "models" / "word_frequency.json"
out.write_text(json.dumps(counts, ensure_ascii=False), encoding="utf-8")
print(f"{len(counts):,} word types, {sum(counts.values()):,} tokens -> {out}")
print("samples:", {w: counts.get(w, 0) for w in ("ọkàn", "ọkà", "òkun", "ọkun")})
