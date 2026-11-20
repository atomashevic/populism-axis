#!/usr/bin/env python3
"""Political DEBATE replacement for the fixed pipeline's BART claim check."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from populism_axis.run_manifest import file_sha256, write_run_manifest

MODEL = 'mlburnham/Political_DEBATE_large_v1.0'
REVISION = '1a3aff1ecb97ad93a6de598f7414b1707de7e7c3'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--names', default='english_dev,english_holdout,agora,eu,usgov_speeches,popbert_en,portugal,antielite_tweets')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    dtype = torch.float16 if args.device.startswith('cuda') else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REVISION, dtype=dtype).eval().to(args.device)
    labels = {str(label).lower(): int(index) for index, label in model.config.id2label.items()}
    if set(labels) != {'entailment', 'not_entailment'}:
        raise ValueError(f'Unexpected native label schema: {labels}')
    hyp_path = args.in_dir / 'hypotheses.json'
    hypotheses = json.loads(hyp_path.read_text())
    hyp_texts = [h['text'] for h in hypotheses]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    for name in filter(None, args.names.split(',')):
        request_path = args.in_dir / f'requests_{name}.parquet'
        out = args.output_dir / f'nli_{name}__political-debate-large.npz'
        manifest_path = args.output_dir / f'nli_{name}__political-debate-large.manifest.json'
        expected = {'script_sha256': file_sha256(Path(__file__)), 'model_revision': REVISION, 'max_length': 320, 'truncation': 'only_first',
                    'dtype': str(dtype), 'native_labels': labels, 'batch_size': args.batch_size}
        if out.exists():
            manifest = json.loads(manifest_path.read_text())
            if any(manifest['extra'].get(k) != v for k, v in expected.items()):
                raise ValueError(f'Existing configuration mismatch: {name}')
            for key, path in [('requests', request_path), ('hypotheses', hyp_path)]:
                if manifest['inputs'][key]['sha256'] != file_sha256(path):
                    raise ValueError(f'Existing input mismatch: {name}/{key}')
            if manifest['extra']['output_sha256'] != file_sha256(out):
                raise ValueError(f'Existing output checksum mismatch: {name}')
            print(f'{name}: validated completed output, skipped', flush=True)
            continue
        requests = pd.read_parquet(request_path)
        premises = requests['premise'].tolist()
        n_req, n_hyp = len(premises), len(hyp_texts)
        order = np.argsort(np.repeat([len(p) for p in premises], n_hyp), kind='stable')
        entailment = np.empty(n_req * n_hyp, dtype=np.float32)
        started = time.monotonic()
        with torch.inference_mode():
            for batch_number, start in enumerate(range(0, len(order), args.batch_size)):
                idx = order[start:start + args.batch_size]
                batch = tokenizer([premises[i // n_hyp] for i in idx], [hyp_texts[i % n_hyp] for i in idx],
                                  padding=True, truncation='only_first', max_length=320, return_tensors='pt').to(args.device)
                entailment[idx] = torch.softmax(model(**batch).logits.float(), dim=-1)[:, labels['entailment']].cpu().numpy()
                if batch_number % 200 == 0:
                    print(f'{name}: {start + len(idx)}/{len(order)} pairs; {time.monotonic()-started:.1f}s', flush=True)
        if not np.isfinite(entailment).all():
            raise FloatingPointError(f'Nonfinite probabilities: {name}')
        temp = out.with_suffix('.partial.npz')
        np.savez_compressed(temp, request_id=requests['request_id'].to_numpy(),
                            entailment=entailment.reshape(n_req, n_hyp),
                            hypothesis_ids=np.asarray([h['hypothesis_id'] for h in hypotheses]))
        temp.replace(out)
        info = dict(expected, pairs=len(order), seconds=round(time.monotonic()-started, 2),
                    output_sha256=file_sha256(out),
                    peak_vram_mb=round(torch.cuda.max_memory_allocated()/2**20, 1) if args.device.startswith('cuda') else None)
        write_run_manifest(args.output_dir, script='scripts/political_debate_nli.py',
                           inputs={'requests': request_path, 'hypotheses': hyp_path}, encoder_id=MODEL,
                           extra=info, filename=manifest_path.name)
        print(name, info, flush=True)

if __name__ == '__main__':
    main()
