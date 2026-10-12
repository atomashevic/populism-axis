"""Pinned three-class entailment with explicit offline controls and cleanup."""
import numpy as np


def validate_execution(device, threads, encoder_batch_size, nli_batch_size, local_files_only):
    import re
    for name, value in [('threads', threads), ('encoder_batch_size', encoder_batch_size), ('nli_batch_size', nli_batch_size)]:
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            raise ValueError(f'{name} must be a positive integer')
    if threads is None or nli_batch_size is None:
        raise ValueError('threads and nli_batch_size must be positive integers')
    if not isinstance(local_files_only, bool):
        raise ValueError('local_files_only must be boolean')
    if not isinstance(device, str) or not re.fullmatch(r'cpu|cuda(?::[0-9]+)?', device):
        raise ValueError('device must be cpu, cuda, or cuda:<index>')
    if device.startswith('cuda'):
        import torch
        if not torch.cuda.is_available() or (':' in device and int(device.split(':')[1]) >= torch.cuda.device_count()):
            raise ValueError(f'CUDA device {device!r} is unavailable')


def preflight_models(model_ids, local_files_only):
    """Check complete cached model snapshots before either inference stage starts."""
    if not local_files_only:
        return
    import json
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    for model_id, revision in model_ids:
        try:
            snapshot = Path(snapshot_download(model_id, revision=revision, local_files_only=True))
            if not (snapshot / 'config.json').is_file():
                raise FileNotFoundError('config.json')
            files = ['model.safetensors', 'pytorch_model.bin']
            weights = any((snapshot / name).is_file() for name in files)
            for name in ['model.safetensors.index.json', 'pytorch_model.bin.index.json']:
                index = snapshot / name
                if index.is_file():
                    shards = json.loads(index.read_text())['weight_map'].values()
                    weights = weights or all((snapshot / shard).is_file() for shard in shards)
            if not weights:
                raise FileNotFoundError('model weights')
            AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)
        except (OSError, ValueError) as error:
            raise OSError(f'Cached model {model_id!r} at revision {revision!r} is missing or incomplete. Cache this pinned revision or explicitly set local_files_only=False.') from error


def run_nli(model_id, premises, hypotheses, device, batch_size, max_length, threads, revision=None, local_files_only=False):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    previous_threads = torch.get_num_threads()
    model = None
    try:
        torch.set_num_threads(threads)
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=local_files_only)
        dtype = torch.float16 if device.startswith('cuda') else torch.float32
        model = AutoModelForSequenceClassification.from_pretrained(model_id, dtype=dtype, revision=revision, local_files_only=local_files_only).eval().to(device)
        labels = {label.lower(): int(index) for index, label in model.config.id2label.items()}
        if set(labels) != {'contradiction', 'neutral', 'entailment'}:
            raise ValueError('Claim check requires contradiction, neutral and entailment model labels')
        n_p, n_h = len(premises), len(hypotheses)
        pair_p, pair_h = np.repeat(np.arange(n_p), n_h), np.tile(np.arange(n_h), n_p)
        order = np.argsort([len(premises[i]) for i in pair_p], kind='stable')
        probs = np.zeros(n_p * n_h, dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(order), batch_size):
                idx = order[start:start + batch_size]
                batch = tokenizer([premises[i] for i in pair_p[idx]], [hypotheses[j] for j in pair_h[idx]], padding=True, truncation='only_first', max_length=max_length, return_tensors='pt').to(device)
                probs[idx] = torch.softmax(model(**batch).logits.float(), dim=-1)[:, labels['entailment']].cpu().numpy()
        if not np.isfinite(probs).all():
            raise FloatingPointError('Nonfinite entailment probabilities')
        return probs.reshape(n_p, n_h), getattr(model.config, '_commit_hash', None)
    finally:
        model = None
        torch.set_num_threads(previous_threads)
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
