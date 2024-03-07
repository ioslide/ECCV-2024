import os
import torch
import argparse
import warnings
import pandas as pd
from conf import cfg
from core.utils import *
from core.model import build_model
from tqdm import tqdm
from setproctitle import setproctitle
import methods
from loguru import logger as log

from robustbench.data import load_cifar100c,load_cifar10c,load_imagenet3dcc,load_imagenetc
from robustbench.utils import clean_accuracy as accuracy
import time
warnings.filterwarnings("ignore")

def eval(cfg):
    log.info(f"==>> Start Eval \n SEED: {cfg.SEED} \n TTA: {cfg.ADAPTER.NAME} \n DATASET: {cfg.CORRUPTION.DATASET} \n OPTIM: {cfg.OPTIM.METHOD} \n BS: {cfg.TEST.BATCH_SIZE} \n MODEL: {cfg.MODEL.ARCH} \n ORDER: {cfg.CORRUPTION.ORDER_NUM} \n SEVERITY: {cfg.CORRUPTION.SEVERITY} \n NOTE: {cfg.NOTE}")

    model = build_model(cfg)
    tta_model = getattr(methods, cfg.ADAPTER.NAME).setup(model, cfg)
    tta_model.cuda()

    if cfg.CORRUPTION.DATASET == "cifar10":
        load_image = load_cifar10c
    elif cfg.CORRUPTION.DATASET == "cifar100":
        load_image = load_cifar100c
    elif cfg.CORRUPTION.DATASET == "imagenet_3dcc":
        load_image = load_imagenet3dcc
    elif cfg.CORRUPTION.DATASET == "imagenet":
        load_image = load_imagenetc
    new_results = {
        "method": cfg.ADAPTER.NAME,
        'dataset':cfg.CORRUPTION.DATASET,
        'model': cfg.MODEL.ARCH,
        'batch_size':cfg.TEST.BATCH_SIZE,
        'seed': cfg.SEED,
        'note': cfg.NOTE,
        'Avg': 0
    }
    for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
        corruption_type_acc = 0
        for severity in [1,2,3,4,5,4,3,2,1]:
            x_test, y_test = load_image(
                cfg.CORRUPTION.NUM_EX,
                severity, 
                cfg.DATA_DIR, 
                False,
                [corruption_type]
            )
            
            acc,each_preds = accuracy(
                model=tta_model, 
                x=x_test.cuda(),
                y=y_test.cuda(), 
                batch_size=cfg.TEST.BATCH_SIZE,
                is_enable_progress_bar=cfg.ENABLE_PROGRESS_BAR
            )
            err = 1. - acc
            log.info(f"[{corruption_type}{severity}]: Acc {acc:.2%} || Error {err:.2%}")
            if severity in new_results.keys():
                new_results[f"{corruption_type}_a_{severity}"] = acc * 100
            else:
                new_results[f"{corruption_type}_{severity}"] = acc * 100

            corruption_type_acc += acc * 100

        corruption_type_acc = corruption_type_acc / 9
        new_results[f"{corruption_type}"] = corruption_type_acc
        new_results['Avg'] += corruption_type_acc

    new_results['Avg'] = new_results['Avg'] / len(cfg.CORRUPTION.TYPE)
    log.info(f"[Avg {severity}]: Acc {new_results['Avg']:.2f} || Error {100-new_results['Avg']:.2f}")

    save_path = f'./results/{cfg.CORRUPTION.DATASET}_{cfg.CORRUPTION.ORDER_NUM}_gradual.csv'

    all_results_df = save(new_results,save_path)
    try:
        send_temp(cfg,new_results)
        try:
            df_to_img(all_results_df,"results/temp.jpg",new_results)
            send_img("results/temp.jpg")
        except:
            pass
        send_file(save_path)
    except Exception as e:
        log.warning(f"Send Msg Failed: {e}")
        send_msg(str(e))
        pass

def main():
    parser = argparse.ArgumentParser("Pytorch Implementation for Test Time Adaptation!")
    parser.add_argument(
        '-acfg',
        '--adapter-config-file',
        metavar="FILE",
        default="",
        help="path to adapter config file",
        type=str)
    parser.add_argument(
        '-dcfg',
        '--dataset-config-file',
        metavar="FILE",
        default="",
        help="path to dataset config file",
        type=str)
    parser.add_argument(
        '-ocfg',
        '--order-config-file',
        metavar="FILE",
        default="",
        help="path to order config file",
        type=str)
    parser.add_argument(
        'opts',
        help='modify the configuration by command line',
        nargs=argparse.REMAINDER,
        default=None)

    args = parser.parse_args()

    if len(args.opts) > 0:
        args.opts[-1] = args.opts[-1].strip('\r\n')
        
    torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(args.adapter_config_file)
    cfg.merge_from_file(args.dataset_config_file)
    if args.order_config_file != "":
        cfg.merge_from_file(args.order_config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    seed_everything(cfg.SEED)

    set_logger(cfg.LOG_DIR,cfg.ADAPTER.NAME)
    ds = cfg.CORRUPTION.DATASET
    adapter = cfg.ADAPTER.NAME
    setproctitle(f"TTA:{ds:>8s}:{adapter:<10s}")


    try:
        eval(cfg)
    
    except Exception as e:
        send_msg(f"Error in TTA {e} \n {cfg.SEED} TTA: {cfg.ADAPTER.NAME} DATASET: {cfg.CORRUPTION.DATASET} BS: {cfg.TEST.BATCH_SIZE} MODEL: {cfg.MODEL.ARCH}  ORDER: {cfg.CORRUPTION.ORDER_NUM} SEVERITY: {cfg.CORRUPTION.SEVERITY} \n {cfg.NOTE}")
        raise


if __name__ == "__main__":
    main()