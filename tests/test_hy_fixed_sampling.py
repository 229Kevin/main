"""HyPyraMamba fixed-count split, independent validation and legacy compatibility."""
import ast
import io
import json
import logging
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from utils import dataset
from utils.utils import metrics

SEEDS = [1, 34, 20031025, 9, 10, 11, 424124, 21213, 123, 29]
SOURCE = Path('/home/zwk/CVPR/HSI/HyPyraMamba/HypraMamba/data_load_operate.py')

class SplitCaptured(Exception):
    def __init__(self, state):
        self.state = state

class HyFixedContracts(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(dataset, 'sample_gt_hypyramamba'),
                        'HyPyraMamba fixed-count sampling has not been implemented')

    def fixture(self):
        return np.concatenate([np.full(n, c, dtype=np.int64)
                               for c, n in enumerate((50, 30, 28, 20))]).reshape(8, 16)

    def test_split_matches_original_source_including_small_classes(self):
        source = ast.parse(SOURCE.read_text())
        method = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'sampling')
        scope = {'np': np}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), 'exec'), scope)
        gt = self.fixture()
        saved = np.random.get_state()
        try:
            for seed in SEEDS:
                np.random.seed(seed)
                tr, va, te, _ = scope['sampling']([.1, .01], [30, 10], gt.ravel() + 1, 4, 1)
                expected = []
                for indices in (tr, va, te):
                    labels = np.full(gt.size, -1, dtype=gt.dtype)
                    labels[indices] = gt.ravel()[indices]
                    expected.append(labels.reshape(gt.shape))
                actual = dataset.sample_gt_hypyramamba(gt, 30, 10, seed)
                for a, e in zip(actual, expected):
                    np.testing.assert_array_equal(a, e)
        finally:
            np.random.set_state(saved)
        for split, counts in zip(actual, ((30, 15, 15, 15), (10, 10, 10, 5), (10, 5, 3, 0))):
            self.assertEqual(tuple(int((split == c).sum()) for c in range(4)), counts)
        coverage = sum((part >= 0).astype(int) for part in actual)
        np.testing.assert_array_equal(coverage, np.ones_like(gt))

    def test_sampling_is_seeded_and_does_not_consume_global_numpy_rng(self):
        gt = self.fixture()
        np.random.seed(123)
        saved = np.random.get_state()
        first = dataset.sample_gt_hypyramamba(gt, 30, 10, 1)
        after = np.random.get_state()
        np.testing.assert_array_equal(saved[1], after[1])
        self.assertEqual(saved[2:], after[2:])
        repeated = dataset.sample_gt_hypyramamba(gt, 30, 10, 1)
        changed = dataset.sample_gt_hypyramamba(gt, 30, 10, 34)
        for a, b in zip(first, repeated): np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(first[0], changed[0]))
        for tr, val in ((0, 10), (30, -1)):
            with self.assertRaises(ValueError):
                dataset.sample_gt_hypyramamba(gt, tr, val, 1)

    def test_fixed_metrics_match_hy_nanmean_and_legacy_metrics_remain_unchanged(self):
        target = np.array([[0, 0, 1, 1]])
        pred = np.array([[0, 1, 1, 1]])
        fixed = metrics(pred, target, n_classes=3, ignore_empty_classes=True)
        self.assertEqual(fixed['Accuracy'], 75)
        self.assertEqual(fixed['AA'], 75)
        self.assertEqual(fixed['Kappa'], 50)
        self.assertTrue(np.isnan(fixed['class acc'][2]))
        legacy = metrics(pred, target, n_classes=2)
        self.assertEqual(legacy['AA'], 75)
        self.assertEqual(legacy['Accuracy'], 75)
        with np.errstate(invalid='ignore'):
            old_empty = metrics(pred, target, n_classes=3)
        self.assertTrue(np.isnan(old_empty['AA']))

    def capture_main(self, *arguments):
        gt = self.fixture()
        image = np.zeros((*gt.shape, 200), dtype=np.float32)
        labels = ['large', 'exact30', 'mowed', 'oats']
        def stop_at_dataset(*args, **kwargs):
            frame = sys._getframe(1)
            try:
                scope = frame.f_locals
                state = {k: scope[k] for k in ('train_gt', 'val_gt', 'test_gt', 'seeds', 'split_tag', 'log_filename')}
                state['opts'] = vars(scope['opts']).copy()
            finally:
                del frame
            raise SplitCaptured(state)
        previous = Path.cwd()
        numpy_state = np.random.get_state(); torch_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        stdout = sys.stdout
        try:
            with tempfile.TemporaryDirectory() as directory:
                (Path(directory) / 'config').symlink_to(PROJECT / 'config', target_is_directory=True)
                os.chdir(directory)
                argv = [str(PROJECT/'main.py'), '--model', 'mamba_shared_psa', '--dataset_name', 'ip', *arguments]
                with patch.object(dataset, 'load_mat_hsi', return_value=(image, gt, labels)), \
                     patch.object(dataset, 'HSIDataset', stop_at_dataset), \
                     patch.object(sys, 'argv', argv), patch.object(sys, 'stdout', io.StringIO()), \
                     patch.object(logging, 'basicConfig'), patch.object(logging.getLogger(), 'addHandler'), \
                     patch.object(logging.getLogger('STDOUT'), 'disabled', True):
                    try:
                        runpy.run_path(str(PROJECT/'main.py'), run_name='__main__')
                    except SplitCaptured as exc:
                        return exc.state
            self.fail('main did not reach dataset construction')
        finally:
            sys.stdout = stdout
            os.chdir(previous)
            np.random.set_state(numpy_state); torch.set_rng_state(torch_state)
            if cuda_states is not None: torch.cuda.set_rng_state_all(cuda_states)

    def test_fixed_main_uses_independent_validation_and_separate_output_tag(self):
        state = self.capture_main('--sampling_mode', 'hy_fixed')
        self.assertEqual(state['opts']['num_run'], 10)
        self.assertEqual(state['seeds'], SEEDS)
        self.assertEqual(state['split_tag'], 'hy_fixed_tr30_val10')
        self.assertIn('_hy_fixed_tr30_val10_', state['log_filename'])
        expected = dataset.sample_gt_hypyramamba(self.fixture(), 30, 10, SEEDS[0])
        for key, split in zip(('train_gt', 'val_gt', 'test_gt'), expected):
            np.testing.assert_array_equal(state[key], split)
        self.assertIsNot(state['val_gt'], state['test_gt'])

    def test_ratio_main_preserves_original_split_validation_and_output_names(self):
        state = self.capture_main('--ratio', '0.1')
        self.assertEqual(state['opts']['sampling_mode'], 'ratio')
        self.assertEqual(state['seeds'], SEEDS)
        self.assertEqual(state['split_tag'], '0.1')
        self.assertNotIn('hy_fixed', state['log_filename'])
        self.assertIs(state['val_gt'], state['test_gt'])
        train, test = dataset.sample_gt(self.fixture(), .1, 1)
        np.testing.assert_array_equal(state['train_gt'], train)
        np.testing.assert_array_equal(state['test_gt'], test)

if __name__ == '__main__':
    unittest.main()
