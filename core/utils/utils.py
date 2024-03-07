import re
import os
import os.path as osp
import warnings
import torch
import random
import errno
import pandas as pd
import numpy as np
from loguru import logger as log

max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min

def save(new_results, path):
    try:
        all_results_df = pd.read_csv(path)
        all_results_df = all_results_df.append(new_results, ignore_index=True)
    except:
        mkdir(osp.dirname(path))
        all_results_df = pd.DataFrame(new_results, index=[0])
    all_results_df.to_csv(path, index=False)
    return all_results_df

def seed_everything(seed):
    if seed is None:
        env_seed = os.environ.get("PL_GLOBAL_SEED")
        if env_seed is None:
            seed = _select_seed_randomly(min_seed_value, max_seed_value)
            log.info(f"No seed found, seed set to {seed}")
        else:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = _select_seed_randomly(min_seed_value, max_seed_value)
                log.info(f"Invalid seed found: {repr(env_seed)}, seed set to {seed}")
    elif not isinstance(seed, int):
        seed = int(seed)
    if not (min_seed_value <= seed <= max_seed_value):
        log.info(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
        seed = _select_seed_randomly(min_seed_value, max_seed_value)
    log.info(f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    return seed

def _select_seed_randomly(min_seed_value, max_seed_value):
    return random.randint(min_seed_value, max_seed_value)  # noqa: S311

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def set_single_gpu(need_gpu_mem=17,spec_gpu=None):
    import gpustat
    import time
    if spec_gpu is None:
        def is_gpu_satisfactory(gpu):
            memory_used = gpu.memory_used
            memory_total = gpu.memory_total
            temperature = gpu.temperature
            if memory_total - memory_used > need_gpu_mem * 1024 and temperature < 70:
                return True
            return False
        gpus = gpustat.GPUStatCollection.new_query()
        while True:
            for gpu in gpus.gpus:
                if is_gpu_satisfactory(gpu):
                    selected_gpu = (gpu.index, gpu.memory_total - gpu.memory_used, gpu.temperature)
                    break
            else:
                time.sleep(10)
                gpus = gpustat.GPUStatCollection.new_query()
                continue
            break

        if selected_gpu:
            select_index, mem_size, temperature = selected_gpu
            os.environ["CUDA_VISIBLE_DEVICES"] = str(select_index)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(spec_gpu)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(spec_gpu)

def check_isfile(fpath):
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile

def get_named_submodule(model, sub_name: str):
    names = sub_name.split(".")
    module = model
    for name in names:
        module = getattr(module, name)
    return module

def set_named_submodule(model, sub_name, value):
    names = sub_name.split(".")
    module = model
    for i in range(len(names)):
        if i != len(names) - 1:
            module = getattr(module, names[i])
        else:
            setattr(module, names[i], value)