#!/usr/bin/env bash
# Corrected RoBERTa sigmoid encode: eight existing inputs, pinned revisions, one job.
set -euo pipefail
OUT="results/roberta-correction/rc-20261002T1415Z-sigmoid"
MREV=9b5badfaab879d33fb69a9a22615175bc9b7e6c8
TREV=722cf37b1afa9454edce342e7895e588b6ff1d59
run() {
  local name="$1"; shift
  echo "# start ${name} $(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)"
  python scripts/external/encode_roberta_custom.py --name "$name" --out-dir "$OUT" \
    --model-revision "$MREV" --tokenizer-revision "$TREV" --batch-size 64 --max-length 128 --device cuda "$@"
  echo "# end ${name} $(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)"
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
}
python -c "import torch,transformers,sys;print('# env',sys.version.split()[0],torch.__version__,torch.version.cuda,transformers.__version__,torch.cuda.get_device_name(0))"
run eu --csv data/benchmarks/eu_sentences.csv --text-column text
run popbert_en_test --csv data/benchmark_popbert/test_en.csv --text-column text_en
run english --csv data/benchmarks/english_gpd_speeches.csv --text-path-column path
run agora --csv data/benchmarks/agora_paragraphs.csv --text-column text
run usgov_speeches --csv data/benchmark_usgov/speeches.csv --text-column text
run antielite_tweets --csv data/external/licht_antielite/tweets_text.csv --text-column text_en_clean
run popbert_en_train --csv data/benchmark_popbert/train_en.csv --text-column text_en
run portugal_en --csv data/benchmark_portugal/paragraphs_en.csv --text-column text_en
echo "# all done $(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)"
