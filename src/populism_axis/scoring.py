"""Proposition-scale fitting and scoring. Public calls create no output files."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import warnings
import numpy as np
import pandas as pd
from .definitions import ScaleDefinition, canonical, populism_preset
from .geometry import unit_rows, normalize
from .contextual_axis import split_windows, aggregate_from_window_scores
from .encoders import get_spec, make_encoder
from .model_revisions import E5_LARGE_V2, BART_LARGE_MNLI, PINNED_REVISIONS
from .nli import run_nli, validate_execution, preflight_models
from .preset import standardize, window_quantities, nonzero_median, top_windows
from .text import clean_speech_text, CLEANING_VERSION


@dataclass(frozen=True)
class ScoreResult:
    scores: pd.DataFrame
    evidence: list
    provenance: dict


def _scale(scale):
    if scale is None or (isinstance(scale, str) and scale == 'populism'):
        return populism_preset()
    if not isinstance(scale, ScaleDefinition):
        raise ValueError("scale must be a ScaleDefinition or 'populism'")
    return scale


def scoring_settings(scale, *, unit=None, pooling=None, clean=None, claim_check=None):
    unit = 'speech' if unit is None else unit
    pooling = ('top' if scale.preset else 'mean') if pooling is None else pooling
    clean = False if clean is None else clean
    claim_check = scale.preset if claim_check is None else claim_check
    if unit not in {'speech', 'paragraph', 'sentence', 'tweet'}:
        raise ValueError('unit must be speech, paragraph, sentence or tweet')
    if pooling not in {'mean', 'top'} or (scale.preset and pooling != 'top'):
        raise ValueError("pooling must be mean or top; populism preset requires top")
    if not isinstance(clean, bool) or not isinstance(claim_check, bool):
        raise ValueError('clean and claim_check must be boolean')
    return {'unit': unit, 'pooling': pooling, 'clean': clean, 'claim_check': claim_check,
            'top_k': 1 if unit in {'sentence', 'tweet'} else 8,
            'encoder': E5_LARGE_V2, 'encoder_revision': PINNED_REVISIONS[E5_LARGE_V2],
            'encoder_template': 'query: {text}', 'encoder_max_length': 256,
            'nli_model': BART_LARGE_MNLI if claim_check else None,
            'nli_revision': PINNED_REVISIONS[BART_LARGE_MNLI] if claim_check else None,
            'nli_max_length': 320 if claim_check else None, 'nli_softmax': 'three_class',
            'window_sentences': 3, 'window_stride': 1, 'sentence_word_cap': 60,
            'cleaning_version': CLEANING_VERSION if clean else None,
            'algorithm': 'preset_exact_v1' if scale.preset else 'custom_float64_v1',
            'hypothesis_selection': 'frozen_v1' if scale.preset else 'axis_alignment_min5_v1',
            'combine': scale.combine, 'softness': .25}


def quantity_names(scale, claim_check):
    if scale.preset:
        return ['PE_axis', 'P_axis', 'E_axis'] + (['PE_nli', 'P_nli', 'E_nli'] if claim_check else [])
    names = ['axis']
    if len(scale.components) > 1:
        names += [f'{c.name}__axis' for c in scale.components]
    if claim_check:
        names += ['support']
        if len(scale.components) > 1:
            names += [f'{c.name}__support' for c in scale.components]
    return names


def score_fields(scale, claim_check):
    if scale.preset:
        return ['axis', 'axis_P', 'axis_E', 'axis_z'] + (['claim_check', 'claim_check_P', 'claim_check_E', 'claim_check_z', 'populism_axis_plus_claim_check', 'P_axis_plus_claim_check', 'E_axis_plus_claim_check'] if claim_check else [])
    names = [scale.name, f'{scale.name}__axis', f'{scale.name}__axis_z']
    if claim_check:
        names += [f'{scale.name}__claim_check', f'{scale.name}__claim_check_z', f'{scale.name}__combined']
    if len(scale.components) > 1:
        for c in scale.components:
            prefix = f'{scale.name}__{c.name}'
            names += [f'{prefix}__axis', f'{prefix}__axis_z']
            if claim_check:
                names += [f'{prefix}__claim_check', f'{prefix}__claim_check_z', f'{prefix}__combined']
    return names


def _inputs(texts, ids, settings):
    if isinstance(texts, (str, bytes)):
        raise ValueError('texts must be a sequence of strings, not a scalar string')
    try:
        texts = list(texts)
    except TypeError as error:
        raise ValueError('texts must be a sequence of strings') from error
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError('texts must contain nonempty strings without missing values')
    if ids is None:
        ids = [str(i) for i in range(len(texts))]
    elif isinstance(ids, (str, bytes)):
        raise ValueError('ids must be a sequence of strings')
    else:
        ids = list(ids)
    if len(ids) != len(texts) or any(not isinstance(i, str) or not i.strip() for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('ids must be unique nonempty strings, one per text')
    windows, docs = [], []
    for d, text in enumerate(texts):
        if settings['clean']:
            text = clean_speech_text(text)[0]
        parts = split_windows(text, 3, 1, max_words=60)
        if not text.strip() or not parts or any(not p.strip() for p in parts):
            raise ValueError(f'Document {ids[d]!r} is empty or unusable after preprocessing')
        windows.extend(parts)
        docs.extend([d] * len(parts))
    input_hash = hashlib.sha256(canonical(list(zip(ids, texts))).encode('utf-8')).hexdigest()
    return ids, windows, np.asarray(docs, dtype=int), input_hash


def _encode(scale, windows, settings, device, threads, encoder_batch_size, local_files_only, axes=None, hypotheses=None):
    spec = get_spec('e5-large-v2', max_length=256, batch_size=encoder_batch_size)
    encode, close = make_encoder(spec, device, num_threads=threads, log_every=0, local_files_only=local_files_only)
    try:
        if axes is None:
            axes, selected = {}, []
            for c in scale.components:
                positive = unit_rows(encode(c.positive))
                contrast = unit_rows(encode(c.contrast))
                try:
                    axes[c.name] = normalize(positive.mean(axis=0) - contrast.mean(axis=0))
                except ValueError as error:
                    raise ValueError(f'Component {c.name!r} has zero or near-zero pole-centroid difference') from error
                if settings['claim_check'] and not scale.preset:
                    alignment = positive @ axes[c.name]
                    order = np.argsort(-alignment, kind='stable')[:min(5, len(c.positive))]
                    selected.extend({'component': c.name, 'anchor_index': int(i), 'hypothesis_id': f'{c.name}:{i}', 'text': c.positive[i], 'alignment': float(alignment[i]), 'selection_rule': 'axis_alignment_min5_v1'} for i in order)
            if settings['claim_check'] and scale.preset:
                from .assets import load_preset_assets
                selected = load_preset_assets()['hypotheses']
            hypotheses = selected
        matrix = np.asarray(encode(windows))
        if not scale.preset:
            matrix = matrix.astype(np.float64)
        if not np.isfinite(matrix).all():
            raise ValueError('Encoder returned nonfinite embeddings')
        resolved = getattr(encode, 'resolved_revision', None)
        return matrix, axes, hypotheses, resolved
    finally:
        close()


def _stats(values, quantity, preset=False):
    mean, sd = float(values.mean()), float(values.std())
    if not np.isfinite([mean, sd]).all() or (not preset and sd <= 1e-12):
        raise ValueError(f'Reference quantity {quantity!r} has invalid or zero variance; required SD must exceed 1e-12')
    return [mean, sd]


def _custom_window(matrix, axes, scale, stats=None):
    mu = matrix.mean(axis=0) if stats is None else np.asarray(stats['mu'], dtype=float)
    normalized = unit_rows(matrix - mu)
    used = {'mu': mu.tolist(), 'proj': {}, 'act': {}}
    z, activations = {}, []
    for c in scale.components:
        projection = normalized @ axes[c.name]
        pair = _stats(projection, c.name) if stats is None else stats['proj'][c.name]
        z[c.name] = (projection - pair[0]) / pair[1]
        used['proj'][c.name] = list(pair)
        if scale.combine == 'conjunction' and len(scale.components) > 1:
            activation = np.logaddexp(0., z[c.name] / .25) ** 2
            factor = nonzero_median(activation) if stats is None else stats['act'][c.name]
            if not np.isfinite(factor) or factor <= 1e-12:
                raise ValueError(f'Invalid activation scale for component {c.name!r}')
            activations.append(activation / factor)
            used['act'][c.name] = factor
    if len(z) == 1:
        axis = next(iter(z.values()))
    elif scale.combine == 'mean':
        axis = np.mean(np.vstack(list(z.values())), axis=0)
    else:
        stacked = np.vstack(activations)
        with np.errstate(divide='ignore'):
            axis = np.exp(np.log(stacked).mean(axis=0))
    if not np.isfinite(axis).all():
        raise ValueError('Nonfinite custom window axis score')
    window = {'axis': axis}
    if len(z) > 1:
        window.update({f'{c}__axis': value for c, value in z.items()})
    return window, z, used


def _axis_stage(matrix, axes, scale, settings, doc_index, n, stats):
    rule = 'mean' if settings['pooling'] == 'mean' else f"k:{settings['top_k']}"
    if scale.preset:
        window, used = window_quantities(matrix, axes, stats)
        z = {c: window[f'{c}_axis'] for c in ('P', 'E')}
        primary = window['PE_axis']
        used['mu'] = used['mu'].tolist()
    else:
        window, z, used = _custom_window(matrix, axes, scale, stats)
        primary = window['axis']
    docs = {q: aggregate_from_window_scores(v, doc_index, n, rule) for q, v in window.items()}
    selected = top_windows(primary, doc_index, n, settings['top_k']) if settings['pooling'] == 'top' else {d: np.flatnonzero(doc_index == d) for d in range(n)}
    component_selected = {c: top_windows(v, doc_index, n, settings['top_k']) if settings['pooling'] == 'top' else selected for c, v in z.items()}
    return docs, primary, z, used, selected, component_selected


def _claim_stage(windows, doc_index, n, selected, hypotheses, scale, settings, device, threads, batch, offline):
    positions = np.concatenate([selected[d] for d in range(n)])
    entail, revision = run_nli(BART_LARGE_MNLI, [windows[p] for p in positions], [h['text'] for h in hypotheses], device, batch, 320, threads, PINNED_REVISIONS[BART_LARGE_MNLI], local_files_only=offline)
    if entail.shape != (len(positions), len(hypotheses)) or not np.isfinite(entail).all() or (entail < 0).any() or (entail > 1).any():
        raise ValueError('Invalid entailment output')
    support, matched = {}, {}
    for c in scale.components:
        cols = [i for i, h in enumerate(hypotheses) if h.get('channel' if scale.preset else 'component') == c.name]
        values = entail[:, cols]
        support[c.name] = values.max(axis=1)
        matched[c.name] = [hypotheses[cols[j]]['hypothesis_id'] for j in values.argmax(axis=1)]
    if scale.preset:
        overall = np.sqrt(support['P'] * support['E'])
    elif len(support) == 1:
        overall = next(iter(support.values()))
    elif scale.combine == 'mean':
        overall = np.mean(np.vstack(list(support.values())), axis=0)
    else:
        with np.errstate(divide='ignore'):
            overall = np.exp(np.log(np.vstack(list(support.values()))).mean(axis=0))
    per_window = {'PE_nli' if scale.preset else 'support': overall}
    if scale.preset or len(support) > 1:
        per_window.update({f'{c}_nli' if scale.preset else f'{c}__support': v for c, v in support.items()})
    docs = {q: pd.Series(v).groupby(doc_index[positions]).mean().reindex(range(n)).fillna(0.).to_numpy() for q, v in per_window.items()}
    return docs, support, matched, {int(p): i for i, p in enumerate(positions)}, revision


def _format(ids, doc_index, docs, doc_stats, scale, settings):
    out = pd.DataFrame({'id': ids, 'n_windows': np.bincount(doc_index, minlength=len(ids))})
    def z(q):
        return standardize(docs[q], tuple(doc_stats[q]))
    if scale.preset:
        out['axis'] = docs['PE_axis']
        out['axis_P'], out['axis_E'] = docs['P_axis'], docs['E_axis']
        out['axis_z'] = z('PE_axis')
        if settings['claim_check']:
            out['claim_check'] = docs['PE_nli']
            out['claim_check_P'], out['claim_check_E'] = docs['P_nli'], docs['E_nli']
            out['claim_check_z'] = z('PE_nli')
            out['populism_axis_plus_claim_check'] = (out['axis_z'] + out['claim_check_z']) / 2
            for c in ('P', 'E'):
                out[f'{c}_axis_plus_claim_check'] = (z(f'{c}_axis') + z(f'{c}_nli')) / 2
    else:
        out[scale.name] = z('axis')
        for prefix, aq, sq in [(scale.name, 'axis', 'support')] + ([(f'{scale.name}__{c.name}', f'{c.name}__axis', f'{c.name}__support') for c in scale.components] if len(scale.components) > 1 else []):
            out[f'{prefix}__axis'], out[f'{prefix}__axis_z'] = docs[aq], z(aq)
            if settings['claim_check']:
                out[f'{prefix}__claim_check'], out[f'{prefix}__claim_check_z'] = docs[sq], z(sq)
                out[f'{prefix}__combined'] = (z(aq) + z(sq)) / 2
    return out


def _evidence(ids, windows, doc_index, primary, z, selected, component_selected, support, matched, positions, scale):
    evidence = []
    for d, identifier in enumerate(ids):
        rows = []
        base = np.flatnonzero(doc_index == d)[0]
        for p in selected[d]:
            if scale.preset:
                row = {'window': windows[p], 'P_z': float(z['P'][p]), 'E_z': float(z['E'][p]), 'axis_window_score': float(primary[p])}
                if support:
                    r = positions[int(p)]
                    row.update({'P_claim': matched['P'][r], 'P_entailment': float(support['P'][r]), 'E_claim': matched['E'][r], 'E_entailment': float(support['E'][r])})
            else:
                components = {c: {'projection_z': float(v[p])} for c, v in z.items()}
                if support:
                    r = positions[int(p)]
                    for c in components:
                        components[c].update({'matched_proposition_id': matched[c][r], 'support': float(support[c][r])})
                row = {'window_index': int(p - base), 'window': windows[p], 'selection_score': float(primary[p]), 'components': components}
            rows.append(row)
        if scale.preset:
            evidence.append({'id': identifier, 'top_windows': rows})
        else:
            evidence.append({'id': identifier, 'windows': rows, 'component_window_indices': {c: [int(p - base) for p in selections[d]] for c, selections in component_selected.items()}})
    return evidence


def _compute(texts, ids, scale, settings, reference, *, device, threads, encoder_batch_size, nli_batch_size, local_files_only, fitting=False):
    ids, windows, doc_index, input_hash = _inputs(texts, ids, settings)
    n = len(ids)
    if (fitting or reference == 'corpus') and n < 30:
        raise ValueError(f'reference fitting/corpus scoring requires at least 30 valid documents; received {n}')
    validate_execution(device, threads, encoder_batch_size, nli_batch_size, local_files_only)
    models = [(E5_LARGE_V2, PINNED_REVISIONS[E5_LARGE_V2])]
    if settings['claim_check']:
        models.append((BART_LARGE_MNLI, PINNED_REVISIONS[BART_LARGE_MNLI]))
        if settings['pooling'] == 'mean':
            warnings.warn('Mean pooling with claim_check=True evaluates all windows and can be substantially more expensive than top pooling.', UserWarning, stacklevel=3)
    preflight_models(models, local_files_only)
    saved = reference.to_dict() if hasattr(reference, 'to_dict') else None
    axes = None if saved is None else {c: np.asarray(v, dtype=float) for c, v in saved['axes'].items()}
    hypotheses = None if saved is None else saved['hypotheses']
    matrix, axes, hypotheses, encoder_revision = _encode(scale, windows, settings, device, threads, encoder_batch_size, local_files_only, axes, hypotheses)
    stats = None if saved is None else saved['window']
    doc_stats = None if saved is None else saved['document']
    if reference == 'frozen':
        from .assets import load_preset_assets
        assets = load_preset_assets()
        stats = {**assets['reference_stats'], 'mu': assets['reference_mean'].astype(np.float64)}
        doc_stats = stats['doc'][f"k:{settings['top_k']}"]
    docs, primary, z, used, selected, component_selected = _axis_stage(matrix, axes, scale, settings, doc_index, n, stats)
    if fitting and scale.preset:
        # Materialize saved mean precision before selecting deployment premises.
        docs, primary, z, used, selected, component_selected = _axis_stage(matrix, axes, scale, settings, doc_index, n, used)
    support, matched, positions, nli_revision = {}, {}, {}, None
    if settings['claim_check']:
        nli_docs, support, matched, positions, nli_revision = _claim_stage(windows, doc_index, n, selected, hypotheses, scale, settings, device, threads, nli_batch_size, local_files_only)
        docs.update(nli_docs)
    if doc_stats is None:
        doc_stats = {q: _stats(v, q, scale.preset) for q, v in docs.items()}
    from .run_manifest import package_versions, package_version
    import platform
    execution = {'package_version': package_version(), 'python': platform.python_version(), 'packages': package_versions(), 'device': device, 'threads': threads, 'encoder_batch_size': encoder_batch_size or get_spec('e5-large-v2').batch_size, 'nli_batch_size': nli_batch_size, 'local_files_only': local_files_only, 'resolved_revisions': {'encoder': encoder_revision, 'nli_model': nli_revision}}
    if fitting:
        from .reference import FittedReference
        payload = {'schema': 'populism_axis.reference.v1', 'scale': scale.to_dict(), 'settings': settings, 'axes': {c: v.tolist() for c, v in axes.items()}, 'window': used, 'document': doc_stats, 'hypotheses': hypotheses, 'inputs': {'sha256': input_hash, 'n_documents': n, 'n_windows': len(windows)}, 'execution': execution}
        fitted = FittedReference(payload)
        reloaded = FittedReference.from_dict(fitted.to_dict())
        # Reapply saved precision to cached embeddings, without model reruns.
        repeat = _axis_stage(matrix, axes, scale, settings, doc_index, n, reloaded.to_dict()['window'])
        for q in repeat[0]:
            if not np.array_equal(repeat[0][q], docs[q]):
                raise RuntimeError(f'Reference serialization changed quantity {q!r}')
        return fitted
    from .run_manifest import build_run_manifest
    from .assets import preset_asset_hashes
    provenance = build_run_manifest(script='populism_axis.score_texts', argv=[], encoder_id=E5_LARGE_V2, extra={'settings': settings, 'execution': execution})
    provenance.update({'definition': scale.to_dict(), 'standardization': {'window': used, 'document': doc_stats}, 'definition_identity': scale.identity, 'reference_identity': reference.identity if saved else None, 'reference': 'fitted' if saved else reference, 'settings': settings, 'execution': execution, 'ordered_input_sha256': input_hash, 'n_documents': n, 'n_windows': len(windows), 'asset_hashes': preset_asset_hashes() if scale.preset else {}, 'hypotheses': hypotheses})
    from .definitions import identity
    provenance['run_hash'] = identity({k: provenance[k] for k in ('definition_identity', 'reference_identity', 'reference', 'settings', 'ordered_input_sha256', 'asset_hashes', 'hypotheses')})
    return ScoreResult(_format(ids, doc_index, docs, doc_stats, scale, settings), _evidence(ids, windows, doc_index, primary, z, selected, component_selected, support, matched, positions, scale), provenance)


def fit_reference(texts, *, scale, ids=None, unit=None, pooling=None, clean=None, claim_check=None, device='cpu', threads=4, encoder_batch_size=None, nli_batch_size=16, local_files_only=True):
    scale = _scale(scale)
    settings = scoring_settings(scale, unit=unit, pooling=pooling, clean=clean, claim_check=claim_check)
    return _compute(texts, ids, scale, settings, 'corpus', device=device, threads=threads, encoder_batch_size=encoder_batch_size, nli_batch_size=nli_batch_size, local_files_only=local_files_only, fitting=True)


def score_texts(texts, *, reference, scale=None, ids=None, unit=None, pooling=None, clean=None, claim_check=None, device='cpu', threads=4, encoder_batch_size=None, nli_batch_size=16, local_files_only=True):
    from .reference import FittedReference
    if isinstance(reference, FittedReference):
        scale = reference.scale if scale is None else _scale(scale)
        if scale.identity != reference.scale.identity:
            raise ValueError('Reference definition mismatch: propositions or component configuration changed')
        for name, value in [('unit', unit), ('pooling', pooling), ('clean', clean), ('claim_check', claim_check)]:
            if value is not None and value != reference.settings[name]:
                raise ValueError(f'Reference was fitted with {name}={reference.settings[name]!r}; requested {value!r}')
        settings = reference.settings
    elif isinstance(reference, str) and reference in {'corpus', 'frozen'}:
        scale = _scale(scale)
        if reference == 'frozen' and not scale.preset:
            raise ValueError('Bundled populism reference cannot score a custom scale; fit a reference for this definition')
        settings = scoring_settings(scale, unit=unit, pooling=pooling, clean=clean, claim_check=claim_check)
    else:
        raise ValueError("reference must be a FittedReference, 'corpus', or 'frozen'")
    return _compute(texts, ids, scale, settings, reference, device=device, threads=threads, encoder_batch_size=encoder_batch_size, nli_batch_size=nli_batch_size, local_files_only=local_files_only)
