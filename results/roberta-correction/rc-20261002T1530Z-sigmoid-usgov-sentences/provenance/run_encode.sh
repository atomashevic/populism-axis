#!/usr/bin/env bash
# Corrected RoBERTa sigmoid encode of the US-governor/candidate sentence diagnostic (4,359 windows), pinned revisions, one job.
set -euo pipefail
OUT="results/roberta-correction/rc-20261002T1530Z-sigmoid-usgov-sentences"
MREV=9b5badfaab879d33fb69a9a22615175bc9b7e6c8
TREV=722cf37b1afa9454edce342e7895e588b6ff1d59
python -c "import torch,transformers,sys;print('# env',sys.version.split()[0],torch.__version__,torch.version.cuda,transformers.__version__,torch.cuda.get_device_name(0))"
echo "# start usgov_sentences $(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)"
python scripts/external/encode_roberta_custom.py --name usgov_sentences --out-dir "$OUT" \
  --model-revision "$MREV" --tokenizer-revision "$TREV" --batch-size 64 --max-length 128 --device cuda \
  --csv data/benchmark_usgov/sentences.csv --text-column text
echo "# end usgov_sentences $(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
