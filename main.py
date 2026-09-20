import logging
import sys
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data
#from torchsummaryX import summary
import os
from utils.dataset import load_mat_hsi, sample_gt, HSIDataset, load_disjoint_data
#from utils.whu_ohs_dataset import WHU_OHS_Patch_Dataset
from utils.utils import split_info_print, metrics, show_results
from utils.scheduler import load_scheduler
import utils.get_cls_map as get_cls_map
from models.get_model import get_model
from train import train, test
import json
import datetime
from pathlib import Path
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"

class StreamToLogger(object):
    """Fake file-like stream object that redirects writes to a logger instance."""
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

def setup_logging(log_filename='output.log'):
    """设置日志记录，同时记录到文件和控制台。"""
    log_parent = os.path.dirname(os.path.abspath(log_filename))
    os.makedirs(log_parent, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        filename=log_filename,
        filemode='a'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger('').addHandler(console_handler)

def build_parser():
    parser = argparse.ArgumentParser(description="run patch-based HSI classification")
    parser.add_argument("--model", type=str, default='asf_rsm_group')
    parser.add_argument("--dataset_name", type=str, default="ip")
    parser.add_argument("--dataset_dir", type=str, default="./datasets")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--patch_size", type=int, default=7)
    parser.add_argument("--num_run", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--ratio", type=float, default=None,
                        help="training ratio (default: 0.05 for IP, 0.1 for other datasets)")
    parser.add_argument("--disjoint", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--metrics-out", type=str, default=None)
    parser.add_argument("--skip-map", action="store_true")
    return parser


def resolve_config_path(opts):
    if opts.config is not None:
        return Path(opts.config)
    return Path("config") / f"{opts.model}_{opts.dataset_name}.json"


def resolve_seeds(opts):
    if opts.seed is not None:
        if opts.num_run not in (None, 1):
            raise ValueError("--seed requires --num_run 1 or omission")
        return [opts.seed]
    defaults = {
        "ip": [1, 34, 20031025, 9, 10, 11, 424124, 21213, 123, 29],
        "whulk": [9, 10, 11, 424124, 21213, 231419123, 123, 1, 34, 20031025],
        "hu2018": [202203, 1, 34, 3, 202202, 202203, 424124, 21213, 123, 29],
    }
    if opts.dataset_name not in defaults:
        raise ValueError(f"No default seeds for dataset {opts.dataset_name!r}; use --seed")
    count = 5 if opts.num_run is None else opts.num_run
    if count < 1 or count > len(defaults[opts.dataset_name]):
        raise ValueError(f"--num_run must be in [1, {len(defaults[opts.dataset_name])}]")
    return defaults[opts.dataset_name][:count]


def json_ready_metrics(run_metrics):
    return {
        "oa": float(run_metrics["Accuracy"]),
        "aa": float(run_metrics["AA"]),
        "kappa": float(run_metrics["Kappa"]),
        "class_accuracy": np.asarray(run_metrics["class acc"]).tolist(),
        "confusion_matrix": np.asarray(run_metrics["Confusion matrix"]).tolist(),
    }


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True, default=_json_default)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _apply_model_patch_size(opts):
    patch_sizes = {
        "ssftt": 9 if opts.dataset_name == "hu2018" else 13,
        "speformer": 7,
        "cnn3d": 13,
        "cnn2d": 13,
        "ssrn": 9,
        "posfree_vit": 7,
        "asf_group": 9,
        "ssmamba": 27,
    }
    if opts.model in patch_sizes:
        opts.patch_size = patch_sizes[opts.model]


def main(argv=None):
    opts = build_parser().parse_args(argv)
    if opts.ratio is None:
        opts.ratio = 0.05 if opts.dataset_name == 'ip' else 0.1
    seeds = resolve_seeds(opts)
    config_path = resolve_config_path(opts)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    with config_path.open(encoding="utf-8") as config_file:
        model_config = json.load(config_file)

    output_dir = Path(opts.output_dir) if opts.output_dir else None
    if output_dir is not None and len(seeds) != 1:
        raise ValueError("--output-dir is a single-job interface; select one --seed")
    now_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "train.log"
        atomic_write_json(output_dir / "config.json", model_config)
    else:
        log_path = Path("logs") / f"{opts.model}_{opts.dataset_name}_{now_time}.log"
    setup_logging(str(log_path))
    sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
    print("model_config:\n{}".format(json.dumps(model_config, indent=4)))
    device = torch.device("cuda:{}".format(opts.device))
    _apply_model_patch_size(opts)
    print("experiments will run on GPU device {}".format(opts.device))
    print("model = {}".format(opts.model))
    print("dataset = {}".format(opts.dataset_name))
    print("dataset folder = {}".format(opts.dataset_dir))
    print("patch size = {}".format(opts.patch_size))
    print("batch size = {}".format(opts.bs))
    print("total epoch = {}".format(opts.epoch))
    print("{} for training and {} testing".format(opts.ratio, 1 - opts.ratio))

    results = []
    for run, seed in enumerate(seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print("running an experiment with the {} model".format(opts.model))
        print("run {} / {} (seed={})".format(run + 1, len(seeds), seed))
        image, gt, labels = load_mat_hsi(opts.dataset_name, opts.dataset_dir)
        height, width = gt.shape
        num_classes = len(labels)
        num_bands = image.shape[-1]
        # get train_gt, val_gt and test_gt
        '''trainval_gt, test_gt = sample_gt(gt, opts.ratio, seeds[run])
        train_gt, val_gt = sample_gt(trainval_gt, 0.5, seeds[run])
        del trainval_gt'''
        if opts.disjoint:
            train_gt, test_gt = load_disjoint_data(opts.dataset_name, opts.dataset_dir)
            train_gt -= 1
            print(len(np.where(train_gt > -1)[0]))
            if opts.dataset_name == 'ip':
                import copy
                all_gt = copy.deepcopy(gt)
                train_gt[all_gt == -1] = -1
                all_gt[train_gt != -1] = -1
                test_gt = all_gt
                del all_gt
            elif opts.dataset_name == 'pu':
                train_gt[gt == -1] = -1
                test_gt -= 1
            print(len(np.where(train_gt > -1)[0]))
        else: 
            train_gt, test_gt = sample_gt(gt, opts.ratio, seed)
        val_gt = test_gt
        #val_gt, _ = sample_gt(test_gt, 0.5, seeds[run])
        train_set = HSIDataset(image, train_gt, patch_size=opts.patch_size, data_aug=True)
        val_set = HSIDataset(image, val_gt, patch_size=opts.patch_size, data_aug=False)
        all_data_set = HSIDataset(image, gt, patch_size=opts.patch_size, data_aug=False)
        train_loader = torch.utils.data.DataLoader(train_set, opts.bs, num_workers=8, pin_memory=True, drop_last=True, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, opts.bs, num_workers=8, pin_memory=True, drop_last=False, shuffle=False)
        all_data_loader = torch.utils.data.DataLoader(all_data_set, opts.bs, num_workers=8, pin_memory=True, drop_last=False, shuffle=False)
        model = get_model(model_config['model'], opts.model, opts.dataset_name, opts.patch_size)
        
        '''if run == 0:
            split_info_print(train_gt, test_gt, labels)
            print("network information:")
            with torch.no_grad():
                summary(model, torch.zeros((1, num_bands, opts.patch_size, opts.patch_size)))'''
        
        model = model.to(device)
        

        total_params = sum(parameter.numel() for parameter in model.parameters())
        if run == 0:
            for name, module in model.named_modules():
                # 计算每个模块的参数量
                num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                # 打印模块名称和参数量
                print(f"{name}: {num_params} parameters")
            model.flops(shape=(1, num_bands, opts.patch_size, opts.patch_size))
        #print("total parameters: {}".format(total_params))'''
        optimizer, scheduler = load_scheduler(model_config['schedule'], opts.model, model)

        criterion = nn.CrossEntropyLoss()
        
        if output_dir is not None:
            model_dir = output_dir / "checkpoint"
        else:
            model_dir = Path("checkpoints") / opts.model / opts.dataset_name / str(opts.ratio) / now_time / str(run)

        train(
            model, optimizer, criterion, train_loader, val_loader,
            opts.epoch, str(model_dir), device, scheduler
        )

        # test the model
        probabilities = test(model, str(model_dir), image, opts.patch_size, num_classes, device)
        prediction = np.argmax(probabilities, axis=-1)
        # computing metrics
        run_results = metrics(prediction, test_gt, n_classes=num_classes)  # only for test set
        results.append(run_results)
        show_results(run_results, label_values=labels)
        run_name = opts.run_name or f"{opts.model}_{opts.dataset_name}_seed{seed}"
        payload = {
            "run_name": run_name,
            "seed": seed,
            "config": str(config_path),
            "ratio": opts.ratio,
            "epochs": opts.epoch,
            "parameter_count": total_params,
            "checkpoint": str(model_dir / "model_best.pth"),
            **json_ready_metrics(run_results),
        }
        metrics_path = (
            Path(opts.metrics_out)
            if opts.metrics_out is not None
            else (output_dir / "metrics.json" if output_dir is not None else None)
        )
        if metrics_path is not None:
            atomic_write_json(metrics_path, payload)
        if run == 0 and not opts.skip_map:
            get_cls_map.get_cls_map(prediction, gt, height, width, opts.dataset_name, opts.ratio, opts.model)
        del model, train_set, train_loader

    if len(seeds) > 1:
        show_results(results, label_values=labels, agregated=True)
    return results


if __name__ == "__main__":
    main()


