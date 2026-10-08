"""Pavia University loading and training-entry defaults."""
import io
import logging
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import savemat
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from utils import dataset


class TrainingStartCaptured(Exception):
    """Stop before allocating datasets, models, or training workers."""

    def __init__(self, state):
        super().__init__('training configuration captured')
        self.state = state


class PaviaLoadingContracts(unittest.TestCase):
    def assert_pavia_load(self, alias):
        # Only the requested alias exists: redirecting up to pu must fail.
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / alias
            folder.mkdir()
            raw = np.array([
                [[0, 1, 2], [3, 4, 5]],
                [[6, 7, 8], [np.nan, 10, 11]],
            ], dtype=np.float32)
            savemat(folder / 'PaviaU.mat', {'paviaU': raw})
            savemat(folder / 'PaviaU_gt.mat', {
                'paviaU_gt': np.array([[0, 1], [9, 4]], dtype=np.uint8),
            })
            with patch.object(sys, 'stdout', io.StringIO()):
                image, gt, labels = dataset.load_mat_hsi(alias, directory)

        # NaN pixels become zero before global normalization and band centering.
        expected_image = np.array([
            [[-9, -8, -7], [3, 4, 5]],
            [[15, 16, 17], [-9, -12, -15]],
        ], dtype=np.float32) / 32
        np.testing.assert_array_equal(image, expected_image)
        self.assertEqual(image.dtype, np.float32)
        np.testing.assert_array_equal(gt, [[-1, 0], [8, -1]])
        self.assertEqual(labels, [
            'Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets',
            'Bare Soil', 'Bitumen', 'Self-Blocking Bricks', 'Shadows',
        ])

    def test_up_reads_its_own_mat_files_with_pavia_preprocessing(self):
        self.assert_pavia_load('up')

    def test_legacy_pu_keeps_the_same_labels_and_preprocessing(self):
        self.assert_pavia_load('pu')


class TrainingEntryContracts(unittest.TestCase):
    def capture_training_start(self, dataset_name, *arguments):
        def stop_at_loader(name, directory):
            frame = sys._getframe(1)
            try:
                scope = frame.f_locals
                state = {
                    'opts': vars(scope['opts']).copy(),
                    'config': scope['model_config'],
                    'config_path': scope['config_dir'],
                    'seeds': list(scope['seeds']),
                    'run': scope['run'],
                    'numpy_seed': int(np.random.get_state()[1][0]),
                    'torch_seed': torch.initial_seed(),
                    'loader_arguments': (name, directory),
                }
            finally:
                del frame
            raise TrainingStartCaptured(state)

        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        previous_directory = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                # Keep real JSON selection while containing main's logs directory.
                (Path(directory) / 'config').symlink_to(PROJECT / 'config', target_is_directory=True)
                os.chdir(directory)
                argv = [str(PROJECT / 'main.py'), '--model', 'basic_spa_spe_mamba',
                        '--dataset_name', dataset_name, *arguments]
                with patch.object(dataset, 'load_mat_hsi', stop_at_loader), \
                     patch.object(sys, 'argv', argv), \
                     patch.object(sys, 'stdout', io.StringIO()), \
                     patch.object(logging, 'basicConfig'), \
                     patch.object(logging.getLogger(), 'addHandler'), \
                     patch.object(logging.getLogger('STDOUT'), 'disabled', True):
                    try:
                        runpy.run_path(str(PROJECT / 'main.py'), run_name='__main__')
                    except TrainingStartCaptured as captured:
                        return captured.state
            self.fail('main.py did not reach dataset loading')
        finally:
            os.chdir(previous_directory)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    def assert_ip_seed_schedule(self, state):
        self.assertEqual(state['seeds'], [
            1, 34, 20031025, 9, 10, 11, 424124, 21213, 123, 29,
        ])
        self.assertEqual(state['run'], 0)
        self.assertEqual(state['numpy_seed'], 1)
        self.assertEqual(state['torch_seed'], 1)

    def test_up_defaults_to_five_percent_and_selects_up_config(self):
        state = self.capture_training_start('up')
        opts = state['opts']
        self.assertEqual(opts['ratio'], 0.05)
        self.assertEqual(opts['patch_size'], 7)
        self.assertEqual(opts['bs'], 64)
        self.assertEqual(opts['epoch'], 300)
        self.assertEqual(opts['num_run'], 10)
        self.assertEqual(state['loader_arguments'], ('up', './datasets'))
        self.assertEqual(state['config_path'], 'config/basic_spa_spe_mamba_up.json')
        self.assertEqual(state['config']['model']['in_chans'], 103)
        self.assertEqual(state['config']['model']['num_classes'], 9)
        self.assert_ip_seed_schedule(state)

    def test_up_explicit_ratio_overrides_the_default(self):
        state = self.capture_training_start('up', '--ratio', '0.02')
        self.assertEqual(state['opts']['ratio'], 0.02)
        self.assert_ip_seed_schedule(state)

    def test_ip_retains_five_percent_and_its_seed_schedule(self):
        state = self.capture_training_start('ip')
        self.assertEqual(state['opts']['ratio'], 0.05)
        self.assertEqual(state['config']['model']['in_chans'], 200)
        self.assertEqual(state['config']['model']['num_classes'], 16)
        self.assert_ip_seed_schedule(state)

    def test_whulk_retains_ten_percent_and_its_seed_schedule(self):
        state = self.capture_training_start('whulk')
        self.assertEqual(state['opts']['ratio'], 0.1)
        self.assertEqual(state['seeds'], [
            9, 10, 11, 424124, 21213, 231419123, 123, 1, 34, 20031025,
        ])
        self.assertEqual(state['numpy_seed'], 9)
        self.assertEqual(state['torch_seed'], 9)


if __name__ == '__main__':
    unittest.main()
