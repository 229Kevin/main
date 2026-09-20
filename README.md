# DualMamba
**TGRS2024 DualMamba: A Lightweight Spectral–Spatial Mamba-Convolution Network for Hyperspectral Image Classification**
[Link](https://ieeexplore.ieee.org/abstract/document/10798573)

## Installation
Follow the installation instructions of the [VMamba](https://github.com/VMamba/VMamba) project to set up the `rs_mamba` environment.

## Dataset Directory Structure
Please organize your dataset as follows:
```
├── datasets
│ ├── ip
│ ├── whulk
│ ├── hu2018
│ ├── pu
```

## Model Training
To train DualMamba, run the following command:
```bash
python main.py --dataset_name ip --model asf_rsm_group

```

## Spatial Mamba S0-S7 Ablation

The spatial ablation uses Indian Pines with a 10% training split, 300 epochs,
and the fixed seeds `1`, `34`, and `20031025`. The eight stages are standalone
configurations under `config/spa_ablation/`:

| Stage | Spatial change |
| --- | --- |
| S0 | Single snake baseline |
| S1 | Shared-parameter bidirectional snake |
| S2 | Shared-parameter four-way cross scan |
| S3 | Outside-in / inside-out ring scan |
| S4 | S3 plus center-aware readout |
| S5 | S3 plus boundary-aware B injection |
| S6 | S5 plus 2D context for Delta/B/C parameters |
| S7 | S6 plus compressed pre-observation state fusion |

Activate the project environment and inspect the complete 24-job matrix before
starting it:

```bash
conda activate LDMT-Mamba
python scripts/run_spa_ablation.py --dry-run
python scripts/run_spa_ablation.py --device 0
```

To run only selected jobs:

```bash
python scripts/run_spa_ablation.py --device 0 --stages S3 S4 --seeds 1
```

Outputs are written to `experiments/spa_ablation_ip_10pct/`. Each stage/seed
directory contains the frozen config, training log, best checkpoint, metrics,
and manifest. The root contains `runs.csv`, `summary.csv`, and `summary.json`
after all requested jobs finish. The runner stops on the first failed job. Run
the same command again to resume: jobs with complete, matching metrics are
skipped, while incomplete jobs are rerun without deleting their artifacts.

For consistency with the existing training pipeline, validation currently uses
the test split (`val_gt = test_gt`). Treat the resulting comparison as an
internal ablation unless a separate validation protocol is introduced.


## Citation
If this code repository is helpful to you, please cite the paper using the BibTeX entry below:
```bibtex
@ARTICLE{dualmamba,
  author={Sheng, Jiamu and Zhou, Jingyi and Wang, Jiong and Ye, Peng and Fan, Jiayuan},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={DualMamba: A Lightweight Spectral–Spatial Mamba-Convolution Network for Hyperspectral Image Classification}, 
  year={2025},
  volume={63},
  number={},
  pages={1-15},
  doi={10.1109/TGRS.2024.3516817}
}
```
