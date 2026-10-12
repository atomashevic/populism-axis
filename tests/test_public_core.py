import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import pandas as pd
from populism_axis import define_scale, fit_reference, score_texts, FittedReference, make_classifier, ThresholdRule, read_scale_csv
from populism_axis import scoring
from populism_axis.definitions import populism_preset
from populism_axis.geometry import unit_rows
from populism_axis.preset import window_quantities, standardize, nonzero_median, top_windows


def vector(text):
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little')
    return np.random.default_rng(seed).normal(size=12).astype(np.float32)


def fake_encoder(*args, **kwargs):
    def encode(texts):
        return np.vstack([vector(t) for t in texts])
    return encode, lambda: None


def fake_nli(model, premises, hypotheses, *args, **kwargs):
    values = np.array([[.05 + .9 * (int.from_bytes(hashlib.sha256((p+h).encode()).digest()[:4], 'little') / (2**32)) for h in hypotheses] for p in premises])
    return values, None


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.scale = define_scale('trust', positive=['Fair institutions.', 'Reliable institutions.'], contrast=['Unfair institutions.'])
        self.texts = [f'Example document {i} discusses institution {i*i}. Citizens express view {i+2}. Officials answer on topic {i+3}. Further evidence for case {i+4}.' for i in range(35)]
        for target, value in [('make_encoder', fake_encoder), ('preflight_models', lambda *a: None), ('run_nli', fake_nli)]:
            patcher = patch.object(scoring, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_definition_csv_and_immutability(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scale.csv'
            path.write_text('component,pole,proposition\ntrust,positive,Fair institutions.\ntrust,positive,Reliable institutions.\ntrust,contrast,Unfair institutions.\n')
            self.assertEqual(read_scale_csv(path, name='trust'), self.scale)
            self.scale.save(Path(directory)/'scale.json')
            self.assertEqual(type(self.scale).load(Path(directory)/'scale.json'), self.scale)
            with self.assertRaises(FileExistsError):
                self.scale.save(Path(directory)/'scale.json')
        with self.assertRaises(ValueError):
            define_scale('trust', positive=['Café', 'Cafe\u0301'], contrast=['Else'])
        with self.assertRaises(ValueError):
            define_scale('trust', positive=[' Yes  indeed '], contrast=['Yes indeed'])
        with self.assertRaises(ValueError):
            define_scale('trust__bad', positive=['Yes'], contrast=['No'])
        with self.assertRaises(ValueError):
            define_scale('trust', components={'a': {'positive':['A'], 'contrast':['B']}, 'b': {'positive':['C'], 'contrast':['D']}})
        with self.assertRaises(Exception):
            self.scale.name = 'new'

    def test_reference_roundtrip_invariance_mean_top_and_support(self):
        for pooling in ('mean', 'top'):
            for claim_check in (False, True):
                with self.subTest(pooling=pooling, claim_check=claim_check):
                    ref = fit_reference(self.texts, scale=self.scale, pooling=pooling, claim_check=claim_check)
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory)/'reference.json'
                        ref.save(path)
                        ref = FittedReference.load(path)
                    batch = score_texts(self.texts, reference=ref)
                    alone = score_texts([self.texts[9]], reference=ref)
                    np.testing.assert_array_equal(batch.scores.iloc[9, 2:].to_numpy(float), alone.scores.iloc[0, 2:].to_numpy(float))
                    self.assertAlmostEqual(batch.scores.trust.mean(), 0., places=12)
                    self.assertAlmostEqual(batch.scores.trust.std(ddof=0), 1., places=12)
                    np.testing.assert_array_equal(batch.scores.trust, batch.scores.trust__axis_z)
                    self.assertEqual(len(batch.evidence[0]['windows']), 2)
                    self.assertEqual(ref.settings['claim_check'], claim_check)
                    snapshot = ref.to_dict()
                    snapshot['settings']['encoder_revision'] = '0'*40
                    with self.assertRaises(ValueError):
                        FittedReference.from_dict(snapshot)
                    with self.assertRaises(ValueError):
                        score_texts(self.texts, reference=ref, unit='sentence')
                    with self.assertRaises(ValueError):
                        score_texts(self.texts, reference=ref, scale=define_scale('trust', positive=['Changed'], contrast=['Unfair institutions.']))

    def test_invalid_inputs_precede_models_and_zero_variance(self):
        with patch.object(scoring, 'make_encoder', side_effect=AssertionError('model called')):
            for texts in ['scalar', [], [None], [' '], ['Valid']]:
                with self.assertRaises(ValueError):
                    fit_reference(texts, scale=self.scale)
            for ids in [['same'] * 35, list(range(35))]:
                with self.assertRaises(ValueError):
                    fit_reference(self.texts, scale=self.scale, ids=ids)
            with self.assertRaises(ValueError):
                score_texts(['one'], reference='frozen', scale=self.scale)
        with self.assertRaisesRegex(ValueError, 'variance'):
            fit_reference(['Identical document.'] * 30, scale=self.scale)
        with patch.object(scoring, 'make_encoder', return_value=(lambda texts: np.ones((len(texts), 12)), lambda: None)):
            with self.assertRaisesRegex(ValueError, 'centroid'):
                fit_reference(self.texts, scale=self.scale)

    def test_multicomponent_combination_and_component_evidence(self):
        components = {'integrity': {'positive':['Honest government'], 'contrast':['Dishonest government']}, 'capacity': {'positive':['Capable government'], 'contrast':['Incapable government']}}
        results = {}
        for combination in ('mean', 'conjunction'):
            scale = define_scale('quality', components=components, combine=combination)
            ref = fit_reference(self.texts, scale=scale, pooling='top', claim_check=True)
            results[combination] = score_texts(self.texts, reference=ref)
            self.assertIn('quality__integrity__axis', results[combination].scores)
            self.assertEqual(set(results[combination].evidence[0]['component_window_indices']), set(components))
            self.assertEqual(len(ref.summary()['hypotheses']), 2)
        self.assertFalse(np.allclose(results['mean'].scores.quality, results['conjunction'].scores.quality))

    def test_hypothesis_selection_min5_stable_ties(self):
        scale = define_scale('test', positive=[f'positive {i}' for i in range(7)], contrast=['contrast'])
        def encoder(*args, **kwargs):
            def encode(texts):
                return np.vstack([np.array([1., 0.]) if t.startswith('positive') else np.array([-1., 0.]) if t == 'contrast' else vector(t)[:2] for t in texts])
            return encode, lambda: None
        with patch.object(scoring, 'make_encoder', encoder):
            ref = fit_reference(self.texts, scale=scale, claim_check=True)
        self.assertEqual([h['anchor_index'] for h in ref.summary()['hypotheses']], list(range(5)))

    def test_threshold_exact_boundary_and_identity(self):
        ref = fit_reference(self.texts, scale=self.scale)
        result = score_texts(self.texts, reference=ref)
        cutoff = float(result.scores.trust.iloc[10])
        rule = make_classifier(ref, threshold=cutoff, below='low', at_or_above='high')
        labels = rule.apply(result)
        self.assertEqual(labels.scores.trust__label.iloc[10], 'high')
        self.assertEqual(labels.scores.trust__label.tolist(), np.where(result.scores.trust >= cutoff, 'high', 'low').tolist())
        pd.testing.assert_frame_equal(labels.scores.iloc[:, :-1], result.scores)
        with tempfile.TemporaryDirectory() as directory:
            rule.save(Path(directory)/'rule.json')
            self.assertEqual(ThresholdRule.load(Path(directory)/'rule.json'), rule)
        with self.assertRaises(ValueError):
            rule.apply(score_texts(self.texts, scale=self.scale, reference='corpus'))
        with self.assertRaises(ValueError):
            rule.apply(labels)
        for threshold in [float('nan'), float('inf'), True]:
            with self.assertRaises(ValueError):
                make_classifier(ref, threshold=threshold, below='low', at_or_above='high')
        with self.assertRaises(ValueError):
            make_classifier(ref, threshold=0., below='same', at_or_above='same')
        with self.assertRaises(ValueError):
            make_classifier(ref, threshold=0., below='low', at_or_above='high', score='trust__combined')

    def test_corpus_provenance_preserves_authored_definition_and_statistics(self):
        result = score_texts(self.texts, scale=self.scale, reference='corpus')
        self.assertEqual(result.provenance['definition'], self.scale.to_dict())
        self.assertEqual(result.provenance['package_version'], '0.1.0')
        self.assertIn('numpy', result.provenance['execution']['packages'])
        self.assertEqual(set(result.provenance['standardization']), {'window', 'document'})
        self.assertEqual(set(result.provenance['standardization']['document']), {'axis'})

    def test_claim_support_preserves_primary_axis_meaning(self):
        axis_only = fit_reference(self.texts, scale=self.scale, claim_check=False)
        supported = fit_reference(self.texts, scale=self.scale, claim_check=True)
        left = score_texts(self.texts, reference=axis_only)
        right = score_texts(self.texts, reference=supported)
        np.testing.assert_array_equal(left.scores.trust, right.scores.trust)
        self.assertIn('trust__combined', right.scores)

    def test_combination_precedes_pooling_and_components_retrieve_independently(self):
        scale = define_scale('quality', components={'a': {'positive':['A'], 'contrast':['B']}, 'b': {'positive':['C'], 'contrast':['D']}}, combine='mean')
        settings = scoring.scoring_settings(scale, pooling='top', unit='sentence')
        matrix = np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
        axes = {'a':np.array([1.,0.]), 'b':np.array([0.,1.])}
        stats = {'mu':[0.,0.], 'proj':{'a':[0.,1.], 'b':[0.,1.]}, 'act':{}}
        docs, primary, z, used, selected, component_selected = scoring._axis_stage(matrix,axes,scale,settings,np.array([0,0,1,1]),2,stats)
        self.assertEqual(docs['axis'][0], .5)
        self.assertEqual(docs['a__axis'][0], 1.)
        self.assertEqual(docs['b__axis'][0], 1.)
        self.assertEqual(selected[0].tolist(), [0])
        self.assertEqual(component_selected['b'][0].tolist(), [1])

    def test_cleanup_on_encoding_failure(self):
        closed = []
        def bad_encoder(*args, **kwargs):
            def encode(texts):
                raise RuntimeError('failure')
            return encode, lambda: closed.append(True)
        with patch.object(scoring, 'make_encoder', bad_encoder):
            with self.assertRaises(RuntimeError):
                fit_reference(self.texts, scale=self.scale)
        self.assertEqual(closed, [True])

    def test_preset_fitted_reference_and_axis_only(self):
        ref = fit_reference(self.texts, scale='populism', claim_check=False)
        result = score_texts(self.texts[:1], reference=ref)
        self.assertEqual(result.scores.columns.tolist(), ['id', 'n_windows', 'axis', 'axis_P', 'axis_E', 'axis_z'])
        self.assertEqual(set(result.evidence[0]), {'id', 'top_windows'})
        with self.assertRaises(ValueError):
            make_classifier(ref, threshold=0, below='low', at_or_above='high')


class LegacyParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.old_reference = (root/'tests'/'fixtures'/'preset_legacy_ba144e99.py').read_text()
        cls.old_scorer = cls.old_reference
        nodes = [n for n in ast.parse(cls.old_reference).body if isinstance(n, ast.FunctionDef) and n.name in {'nonzero_median','standardize','window_quantities'}]
        cls.legacy = {'np':np, 'unit_rows':unit_rows, 'CH':('P','E'), 'SOFTNESS':.25}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<legacy>', 'exec'), cls.legacy)

    def test_native_float32_corpus_frozen_zero_sd_and_ties(self):
        rng = np.random.default_rng(931)
        axes = {c: vector(c).astype(float)/np.linalg.norm(vector(c).astype(float)) for c in ('P','E')}
        matrix = (1000 + rng.normal(size=(90,12))).astype(np.float32)
        old, used = self.legacy['window_quantities'](matrix, axes, None)
        for stats in [None, used, {'mu':used['mu'].astype(float), 'proj':{'P':[0.,0.], 'E':[0.,0.]}, 'act':{'P':1.,'E':1.}}]:
            expected, _ = self.legacy['window_quantities'](matrix, axes, stats)
            actual, _ = window_quantities(matrix, axes, stats)
            for q in expected:
                np.testing.assert_array_equal(actual[q], expected[q])
        nodes = [n for n in ast.parse(self.old_scorer).body if isinstance(n, ast.FunctionDef) and n.name == 'top_windows']
        env = {'np':np}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<legacy>','exec'), env)
        values = np.array([3.,3.,1.,4.,4.,4.]); docs=np.array([0,0,0,1,1,1])
        for k in [1,8]:
            for d, rows in env['top_windows'](values,docs,2,k).items():
                np.testing.assert_array_equal(rows, top_windows(values,docs,2,k)[d])

    def test_all_preset_columns_match_legacy_formatter(self):
        n = 4
        rng = np.random.default_rng(943)
        docs = {q:rng.random(n) for q in ['PE_axis','P_axis','E_axis','PE_nli','P_nli','E_nli']}
        ids=['001','NA','ž','last']; doc_index=np.array([0,0,1,2,3,3])
        main = next(n for n in ast.parse(self.old_scorer).body if isinstance(n,ast.FunctionDef) and n.name == 'main')
        start = next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ref' for t in n.targets))
        end = next(i for i,n in enumerate(main.body[start:],start) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='to_csv')
        scale=populism_preset()
        for enabled in [False,True]:
            for frozen in [False,True]:
                ref_stats={q:[float(v.mean()),float(v.std())] for q,v in docs.items()}
                stats={'doc':{'k:8':ref_stats}} if frozen else None
                env={'pd':pd,'np':np,'docs':docs,'ids':ids,'n':n,'doc_index':doc_index,'args':SimpleNamespace(no_claim_check=not enabled),'CH':('P','E'),'standardize':self.legacy['standardize'],'stats':stats,'rule':'k:8'}
                exec(compile(ast.Module(body=main.body[start:end],type_ignores=[]),'<legacy>','exec'),env)
                actual=scoring._format(ids,doc_index,docs,ref_stats,scale,scoring.scoring_settings(scale,claim_check=enabled))
                pd.testing.assert_frame_equal(actual,env['out'],check_exact=True)

    def test_selected_preset_evidence_matches_legacy(self):
        scale = populism_preset()
        ids = ['001','NA']; windows = ['one','two','three','four']
        doc_index = np.array([0,0,1,1]); primary = np.array([2.,2.,3.,1.])
        z = {'P':np.array([.1,.2,.3,.4]), 'E':np.array([-.1,-.2,-.3,-.4])}
        tops = top_windows(primary,doc_index,2,1)
        support = {'P':np.array([.7,.8]), 'E':np.array([.2,.3])}
        best = {'P':['P00','P01'], 'E':['E00','E01']}
        positions = {int(p):i for i,p in enumerate(np.concatenate(list(tops.values())))}
        main = next(n for n in ast.parse(self.old_scorer).body if isinstance(n,ast.FunctionDef) and n.name == 'main')
        start = next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='evidence' for t in n.targets))
        for enabled in [False,True]:
            env = {'np':np,'n':2,'tops':tops,'windows':windows,'ids':ids,'args':SimpleNamespace(no_claim_check=not enabled),'window':{'P_axis':z['P'],'E_axis':z['E'],'PE_axis':primary},'best':best,'per_window':{'P_nli':support['P'],'E_nli':support['E']}}
            exec(compile(ast.Module(body=main.body[start:-1],type_ignores=[]),'<legacy>','exec'),env)
            actual = scoring._evidence(ids,windows,doc_index,primary,z,tops,{},support if enabled else {},best,positions,scale)
            self.assertEqual(actual,env['evidence'])
