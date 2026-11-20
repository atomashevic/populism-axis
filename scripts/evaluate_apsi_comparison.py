#!/usr/bin/env python3
"""Prepare seven benchmark corpora and evaluate pinned external APSI comparisons.

Text and NLI requests are written only as ignored parquet caches. Existing scores
are read unchanged. Every pair is evaluated on its shared non-abstaining rows.
Intervals resample texts within each benchmark's clusters (leaders, state-terms,
politicians, speakers, countries); PopBERT has none, so its sentences are resampled.
EU macro-F1 intervals condition on the original LOSO predictions.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmarks import load_documents
from evaluate_benchmarks import log_access
from evaluate_eu_paper_protocol import best_f1_threshold, window_max
from evaluate_nli import zscore
from populism_axis.resampling import sampler
from populism_axis.run_manifest import write_run_manifest
from populism_axis.text import clean_speech_text

SYSTEM = 'PE__frozen__soft'
DATASETS = ('english_dev', 'english_holdout', 'agora', 'eu', 'usgov_speeches',
            'popbert_en', 'portugal', 'antielite_tweets')


def aligned(frame: pd.DataFrame, scores: pd.DataFrame, key: str = 'id') -> None:
    if frame[key].astype(str).tolist() != scores[key].astype(str).tolist():
        raise ValueError(f'Score order mismatch on {key}')


def pool_nli(requests: pd.DataFrame, entailment: np.ndarray, hypotheses: list,
             n: int, k: int) -> dict[str, np.ndarray]:
    if entailment.shape != (len(requests), len(hypotheses)):
        raise ValueError('NLI shape mismatch')
    if not np.isfinite(entailment).all() or (entailment < 0).any() or (entailment > 1).any():
        raise ValueError('NLI entailment probabilities must be finite and within [0,1]')
    win = requests[['doc_index', 'window_rank']].copy()
    for c in ('P', 'E'):
        cols = [i for i, h in enumerate(hypotheses) if h['channel'] == c]
        win[c] = entailment[:, cols].max(axis=1)
    win['PE'] = np.sqrt(win.P * win.E)
    selected = win[win.window_rank < k]
    return {c: selected.groupby('doc_index')[c].mean().reindex(range(n)).fillna(0).to_numpy()
            for c in ('P', 'E', 'PE')}


def load_entailment(path: Path, requests: pd.DataFrame) -> np.ndarray:
    payload = np.load(path)
    if not np.array_equal(payload['request_id'], requests.request_id.to_numpy()):
        raise ValueError(f'{path}: request ids differ')
    if 'entailment' in payload:
        return payload['entailment'].astype(float)
    if payload['probs'].shape[-1] != 3:
        raise ValueError('Noncanonical probabilities require explicit entailment matrix')
    return payload['probs'][:, :, 2].astype(float)


def prepare(work: Path, reason: str, access_log: Path | None = None) -> None:
    if not reason:
        raise ValueError('An explicit comparator access reason is required')
    work.mkdir(parents=True, exist_ok=True)
    access = {'timestamp_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'script': 'scripts/evaluate_apsi_comparison.py', 'stage': 'prepare',
              'datasets': list(DATASETS), 'reason': reason, 'selection': 'none'}
    if access_log is None:
        log_access(access)
    else:
        access_log.parent.mkdir(parents=True, exist_ok=True)
        entries = json.loads(access_log.read_text()) if access_log.exists() else []
        entries.append(access)
        access_log.write_text(json.dumps(entries, indent=2) + '\n')
    core = load_documents(ROOT / 'data/benchmarks')
    split = pd.read_csv(ROOT / 'data/benchmarks/english_split.csv').set_index('id')['split']
    frames, texts, sources = {}, {}, {}
    inputs = {}
    for part in ('dev', 'holdout'):
        d = 'english_' + part
        keep = core['english']['frame'].id.map(split).eq(part).to_numpy()
        frames[d] = core['english']['frame'].loc[keep].reset_index(drop=True)
        texts[d] = [clean_speech_text(t)[0] for t, use in zip(core['english']['texts'], keep) if use]
        sources[d] = ROOT / 'results/step6_nli'
    for d in ('agora', 'eu'):
        frames[d] = core[d]['frame'].rename(columns={'row_id': 'id'})
        texts[d] = core[d]['texts']; sources[d] = ROOT / 'results/step6_nli'
    specs = {'usgov_speeches': ('usgov', 'data/benchmark_usgov/speeches.csv', 'text'),
             'portugal': ('portugal', 'data/benchmark_portugal/paragraphs_en.csv', 'text_en'),
             'antielite_tweets': ('antielite_tweets', 'data/external/licht_antielite/tweets_text.csv', 'text_en_clean')}
    for d, (directory, file, col) in specs.items():
        source = ROOT / file; frame = pd.read_csv(source)
        inputs[d + '_text'] = source
        if d == 'antielite_tweets':
            labels = pd.read_csv(ROOT / 'data/benchmark_antielite_tweets/tweets_labels.csv')
            aligned(labels, frame, 'row_id')
            frame = frame.rename(columns={'id': 'source_id', 'row_id': 'id'})
        frames[d] = frame; texts[d] = frame[col].astype(str).tolist()
        sources[d] = ROOT / 'results/external' / directory
    parts = []
    for part in ('train', 'test'):
        path = ROOT / f'data/benchmark_popbert/{part}_en.csv'
        inputs['popbert_' + part] = path
        parts.append(pd.read_csv(path).assign(split=part))
    frames['popbert_en'] = pd.concat(parts, ignore_index=True)
    texts['popbert_en'] = frames['popbert_en'].text_en.astype(str).tolist()
    sources['popbert_en'] = ROOT / 'results/external/popbert_en'
    hypotheses_path = ROOT / 'results/step6_nli/hypotheses.json'
    hypotheses = json.loads(hypotheses_path.read_text())
    shutil.copyfile(hypotheses_path, work / 'hypotheses.json')
    documents, inventory = [], {}
    for d in DATASETS:
        frame = frames[d]; src = sources[d]
        req_path = src / f'requests_{d}.parquet'
        requests = pd.read_parquet(req_path)
        if requests.doc_index.max() >= len(frame):
            raise ValueError(f'{d}: request document index out of bounds')
        expected = frame.id.astype(str).to_numpy()[requests.doc_index.to_numpy()]
        if not np.array_equal(expected, requests.doc_id.astype(str).to_numpy()):
            raise ValueError(f'{d}: requests mismatch frame ids')
        if json.loads((src / 'hypotheses.json').read_text()) != hypotheses:
            raise ValueError(f'{d}: claim hypotheses differ')
        shutil.copyfile(req_path, work / req_path.name)
        k = 1 if d in ('eu', 'popbert_en', 'antielite_tweets') else 8
        baseline = pd.DataFrame({'id': frame.id.astype(str)})
        if d in ('english_dev', 'english_holdout', 'agora', 'eu'):
            short = 'english' if d.startswith('english') else d
            held = d in ('english_holdout', 'eu')
            axis_path = ROOT / ('results/final/B4_axis' if held else 'results/step4_windows_topk/grid/spw3_k8') / f'scores_{short}.csv'
            final_path = ROOT / ('results/final/B6_axis_plus_nli' if held else 'results/step6_nli/combined_k8') / f'scores_{short}.csv'
            axis = pd.read_csv(axis_path); final = pd.read_csv(final_path)
            aligned(frame, axis); aligned(frame, final)
            baseline['axis_PE'] = axis[SYSTEM]; baseline['ours_PE'] = final[SYSTEM]
            if d == 'eu':
                proj_path = ROOT / 'results/final/B4_axis/projections_eu.npz'
                proj = np.load(proj_path)
                bart_path = src / 'nli_eu__bart-large-mnli.npz'
                bart = pool_nli(requests, load_entailment(bart_path, requests), hypotheses, len(frame), 1)
                for c in ('P', 'E'):
                    baseline['axis_' + c] = window_max(proj['doc_index'].astype(int), proj[c], len(frame))
                    baseline['ours_' + c] = (zscore(baseline['axis_' + c].to_numpy()) + zscore(bart[c])) / 2
                inputs['eu_projections'] = proj_path; inputs['eu_bart'] = bart_path
        else:
            final_path = src / ('final_scores_speeches.csv' if d == 'usgov_speeches' else 'final_scores.csv')
            final = pd.read_csv(final_path)
            if d == 'antielite_tweets':
                final = final.rename(columns={'id': 'source_id', 'row_id': 'id'})
            aligned(frame, final)
            if d == 'usgov_speeches':
                baseline['axis_PE'] = final.B4_axis; baseline['ours_PE'] = final.B6_axis_plus_nli
            else:
                for c in ('P', 'E', 'PE'):
                    if f'{c}_axis_plus_nli' in final:
                        baseline['axis_' + c] = final[f'{c}_axis']
                        baseline['ours_' + c] = final[f'{c}_axis_plus_nli']
            axis_path = final_path
        inputs[d + '_axis'] = axis_path; inputs[d + '_baseline'] = final_path
        inputs[d + '_requests'] = req_path
        baseline.to_parquet(work / f'baseline_{d}.parquet', index=False)
        drop = [c for c in frame if c.startswith('text') or c in ('sentence', 'translated_sentence')]
        frame.drop(columns=drop).to_parquet(work / f'labels_{d}.parquet', index=False)
        documents.append(pd.DataFrame({'dataset': d, 'id': frame.id.astype(str), 'text': texts[d]}))
        inventory[d] = {'n': len(frame), 'requests': len(requests), 'top_k': k}
    pd.concat(documents, ignore_index=True).to_parquet(work / 'documents.parquet', index=False)
    (work / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    for f in ('english_gpd_speeches', 'english_split', 'agora_paragraphs', 'eu_sentences'):
        inputs[f] = ROOT / f'data/benchmarks/{f}.csv'
    inputs['documents'] = work / 'documents.parquet'; inputs['hypotheses'] = hypotheses_path
    write_run_manifest(work, script='scripts/evaluate_apsi_comparison.py', inputs=inputs,
                       extra={'stage': 'prepare', 'reason': reason, 'inventory': inventory})
    print(json.dumps(inventory, indent=2))


def metric(name: str, y: np.ndarray, score: np.ndarray) -> float:
    if name == 'macro_f1':
        return float(np.mean([f1_score(y[:, i], score[:, i], zero_division=0) for i in range(3)]))
    if len(np.unique(y)) < 2:
        return float('nan')
    if name == 'spearman':
        return float(spearmanr(y, score).statistic) if len(np.unique(score)) > 1 else float('nan')
    return float(roc_auc_score(y, score))


def comparison(y, left, right, groups, name, draws, seed, scheme='within'):
    finite = np.isfinite(left) & np.isfinite(right)
    keep = finite.all(axis=1) if finite.ndim == 2 else finite
    y, left, right, groups = y[keep], left[keep], right[keep], groups[keep]
    unique = np.unique(groups)
    draw = sampler(groups, scheme) if len(unique) else lambda rng: np.array([], int)
    rng = np.random.default_rng(seed); samples = []
    for _ in range(draws):
        ix = draw(rng)
        if len(ix):
            a, b = metric(name, y[ix], left[ix]), metric(name, y[ix], right[ix])
            if np.isfinite(a) and np.isfinite(b):
                samples.append((a, b, a-b))
    a = metric(name, y, left) if len(y) else float('nan')
    b = metric(name, y, right) if len(y) else float('nan')
    ci = np.percentile(samples, [2.5, 97.5], axis=0).T.tolist() if samples else [[None, None]] * 3
    return {'n_matched': len(y), 'n_clusters': len(unique), 'left': a, 'right': b,
            'delta': a-b, 'left_ci95': ci[0], 'right_ci95': ci[1], 'delta_ci95': ci[2],
            'valid_draws': len(samples), 'requested_draws': draws, 'resampling': scheme}


def loso_predictions(frame, e, p):
    code = frame.pop_code.to_numpy(int)
    truth = np.column_stack((np.isin(code, [1, 3]), np.isin(code, [2, 3]))).astype(int)
    available = np.isfinite(e) & np.isfinite(p)
    pred = np.full((len(frame), 3), np.nan)
    for speaker in frame.speaker.unique():
        train = available & frame.speaker.ne(speaker).to_numpy()
        test = available & frame.speaker.eq(speaker).to_numpy()
        if not train.any():
            continue
        for c, vals in enumerate((e, p)):
            cut = best_f1_threshold(vals[train], truth[train, c])
            pred[test, c + 1] = (vals[test] > cut).astype(int)
        pred[test, 0] = (pred[test, 1:].sum(axis=1) == 0).astype(int)
    truth = np.column_stack(((truth.sum(axis=1) == 0).astype(int), truth))
    return truth, pred


def tasks(d, f):
    all_rows = np.ones(len(f), bool)
    if d.startswith('english'):
        return [('holistic', 'spearman', 'PE', f.source_score.to_numpy(float), f.leader.to_numpy(), all_rows)]
    if d == 'agora':
        return [('holistic', 'spearman', 'PE', f.populism_human.to_numpy(float), f.politician.to_numpy(), all_rows)]
    if d == 'usgov_speeches':
        return [(s, 'spearman', 'PE', f.score.to_numpy(float), f.group.to_numpy(),
                 all_rows if s == 'all_pooled' else f.subset.eq(s).to_numpy())
                for s in ('governors_2010_2018', 'governors_2018_2022', 'presidential_2016', 'all_pooled')]
    if d == 'eu':
        return [('both_vs_neither', 'auroc', 'PE', f.pop_code.eq(3).to_numpy(int), f.speaker.to_numpy(), f.pop_code.isin([0, 3]).to_numpy())]
    if d == 'antielite_tweets':
        return [('anti_elitism', 'auroc', 'E', f.y_general.to_numpy(int), f.country.to_numpy(), f.split.eq('test').to_numpy())]
    # PopBERT has no speaker identifiers: one stratum, within which the sentences are resampled
    groups = f.actor.to_numpy() if d == 'portugal' else np.full(len(f), 'all')
    mask = all_rows if d == 'portugal' else f.split.eq('test').to_numpy()
    return [(label, 'auroc', c, f[col].to_numpy(int), groups, mask)
            for label, c, col in [('anti_elitism', 'E', 'elite_gold'), ('people_centrism', 'P', 'centr_gold')]]


def evaluate(args):
    work = args.work_dir; out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((work / 'inventory.json').read_text())
    hypotheses = json.loads((work / 'hypotheses.json').read_text())
    ap = pd.concat([pd.read_csv(p, dtype={'id': str}).assign(model_size=s)
                    for s, p in [('large', args.apsi_large), ('base', args.apsi_base)]], ignore_index=True)
    if ap.duplicated(['dataset', 'id', 'model_size', 'mode']).any():
        raise ValueError('Duplicate APSI score rows')
    results = []; numeric = []; inputs = {'evaluation_script': Path(__file__),
                                         'hypotheses': work / 'hypotheses.json', 'inventory': work / 'inventory.json',
                                         'apsi_large': args.apsi_large, 'apsi_base': args.apsi_base,
                                         'prepare_manifest': work / 'run_manifest.json'}
    for d in DATASETS:
        for stem in ('labels', 'baseline', 'requests'):
            inputs[d + '_' + stem] = work / f'{stem}_{d}.parquet'
        f = pd.read_parquet(work / f'labels_{d}.parquet'); b = pd.read_parquet(work / f'baseline_{d}.parquet')
        req = pd.read_parquet(work / f'requests_{d}.parquet')
        path = args.political_nli_dir / f'nli_{d}__political-debate-large.npz'
        inputs[d + '_nli'] = path
        nli = pool_nli(req, load_entailment(path, req), hypotheses, len(f), inventory[d]['top_k'])
        channels = [c for c in ('P', 'E', 'PE') if 'ours_' + c in b]
        systems = {'ours': {c: b['ours_' + c].to_numpy(float) for c in channels},
                   'ours_political_debate': {c: (zscore(b['axis_' + c].to_numpy(float)) + zscore(nli[c])) / 2 for c in channels}}
        for size in ('large', 'base'):
            for mode in ('native', 'chunk'):
                rows = ap[(ap.dataset == d) & (ap.model_size == size) & (ap['mode'] == mode)]
                if set(rows.id) != set(b.id.astype(str)):
                    raise ValueError(f'APSI {d}/{size}/{mode}: missing or extra ids')
                vals = rows.set_index('id').score.reindex(b.id.astype(str)).to_numpy(float)
                systems[f'apsi_{size}_{mode}'] = {c: vals for c in channels}
        for name, scores in systems.items():
            for c, vals in scores.items():
                numeric.append(pd.DataFrame({'dataset': d, 'id': b.id, 'system': name, 'channel': c, 'score': vals}))
        specs = tasks(d, f)
        for task, met, c, y, groups, mask in specs:
            vectors = {name: s[c][mask] for name, s in systems.items()}
            add_comparisons(results, d, task, met, y[mask], groups[mask], vectors, args)
        if d == 'eu':
            pairs = [('ours_political_debate', 'ours')]
            pairs += [(a, r) for a in systems if a.startswith('apsi_')
                      for r in ('ours', 'ours_political_debate')]
            for left, right in pairs:
                shared = np.logical_and.reduce([np.isfinite(systems[name][c])
                                                for name in (left, right) for c in ('E', 'P')])
                predictions = {}
                for name in (left, right):
                    e = np.where(shared, systems[name]['E'], np.nan)
                    p = np.where(shared, systems[name]['P'], np.nan)
                    truth, predictions[name] = loso_predictions(f, e, p)
                add_comparisons(results, d, 'original_protocol_loso', 'macro_f1', truth,
                                f.speaker.to_numpy(), predictions, args, pairs=[(left, right)])
                record = results[-1]
                record['shared_threshold_fit_n'] = int(shared.sum())
                record['n_by_pop_code'] = {str(c): int(f.pop_code.eq(c).sum()) for c in range(4)}
                record['matched_by_pop_code'] = {str(c): int((f.pop_code.eq(c).to_numpy() & shared).sum()) for c in range(4)}
                for side, name in [('left', left), ('right', right)]:
                    raw = np.isfinite(systems[name]['E']) & np.isfinite(systems[name]['P'])
                    record[side + '_evaluation_coverage'] = record[side + '_coverage']
                    record[side + '_n_scored'] = int(raw.sum())
                    record[side + '_coverage'] = float(raw.mean())
    pd.concat(numeric, ignore_index=True).to_csv(out / 'scores.csv', index=False)
    pd.DataFrame(results).to_json(out / 'comparisons.json', orient='records', indent=2)
    pd.DataFrame(results).to_csv(out / 'comparisons.csv', index=False)
    write_run_manifest(out, script='scripts/evaluate_apsi_comparison.py', inputs=inputs,
                       extra={'draws': args.draws, 'seed': args.seed, 'resampling': 'texts within clusters',
                              'eu_macro_f1_intervals': 'sentences resampled within speakers, conditional on fixed LOSO predictions',
                              'apsi_component_mapping': 'same overall score, no hypothesis subsets'})
    print(f'Wrote {len(results)} paired comparisons to {out}')


def add_comparisons(results, dataset, task, met, y, groups, vectors, args, pairs=None):
    if pairs is None:
        pairs = [('ours_political_debate', 'ours')]
        pairs += [(a, r) for a in vectors if a.startswith('apsi_') for r in ('ours', 'ours_political_debate')]
    for left, right in pairs:
        l, r = vectors[left], vectors[right]
        finite = lambda v: np.isfinite(v).all(axis=1) if v.ndim == 2 else np.isfinite(v)
        record = {'dataset': dataset, 'task': task, 'metric': met, 'left_system': left, 'right_system': right,
                  'n_eligible': len(y), 'left_n_scored': int(finite(l).sum()), 'right_n_scored': int(finite(r).sum()),
                  'left_coverage': float(finite(l).mean()), 'right_coverage': float(finite(r).mean()),
                  'interval_scope': 'fixed_LOSO_predictions' if met == 'macro_f1' else 'resampled_scores'}
        if met == 'auroc':
            record.update({'n_positive': int(y.sum()),
                           'left_positive_scored': int(y[finite(l)].sum()),
                           'right_positive_scored': int(y[finite(r)].sum())})
        record.update(comparison(y, l, r, groups, met, args.draws, args.seed))
        results.append(record)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'evaluate'))
    p.add_argument('--work-dir', type=Path, default=ROOT / 'results/apsi_comparison/cache')
    p.add_argument('--out-dir', type=Path, default=ROOT / 'results/apsi_comparison')
    p.add_argument('--access-reason', default='')
    p.add_argument('--access-log', type=Path, help='Optional separate access log for clean run snapshots')
    p.add_argument('--apsi-large', type=Path)
    p.add_argument('--apsi-base', type=Path)
    p.add_argument('--political-nli-dir', type=Path)
    p.add_argument('--draws', type=int, default=4000)
    p.add_argument('--seed', type=int, default=20260923)
    args = p.parse_args()
    if args.stage == 'prepare':
        prepare(args.work_dir, args.access_reason, args.access_log)
    else:
        if not all((args.apsi_large, args.apsi_base, args.political_nli_dir)):
            p.error('evaluate requires --apsi-large, --apsi-base and --political-nli-dir')
        evaluate(args)


if __name__ == '__main__':
    main()
