import sys
from pathlib import Path

import numpy as np
import torch

from trainer import Trainer, PreLoader, EarlyStopper
from models import MLP2
from pildataset import PILDataSet
from utils import last_epoch


import random
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

torch.use_deterministic_algorithms(True)
def reset_random_seeds(seed_value=1):
  os.environ['PYTHONHASHSEED']=str(seed_value)
  torch.manual_seed(0)
  np.random.seed(seed_value)
  random.seed(seed_value)


hyper_params = [
    ("all", "SGD", [0.01, 0.01, 0.01]),
    ("FC1", "SGD", [0.01,    0,    0]),
    ("FC2", "SGD", [   0, 0.01,    0]),
    
    ("all", "ADAM", [0.001, 0.001, 0.001]),
    ("FC1", "ADAM", [0.001,     0,     0]),
    ("FC2", "ADAM", [    0, 0.001,     0]),
]


def whiten(m, LR):
  # Whiten the non-training layers, i.e. set all singular values to 1.
  for c, lr in zip(m.children(), LR):
    if lr == 0:
      with torch.no_grad():
        U, _, V = torch.linalg.svd(c.weight, full_matrices=False)
        c.weight = torch.nn.Parameter( U @ V )
        c.bias[:] = 1.


def main(DS, RUNS, LR_search, SCALES, WHITEN, C=1, H=28, W=28, RESTART=False):
  TRAIN = PILDataSet(True,  DS=DS)
  TEST  = PILDataSet(False, DS=DS)

  # Ensure that a different initialization exists for each seed.
  for run in range(RUNS):
    if not Trainer.save_dir(run, 0, "SETOL/TEST").exists():
      reset_random_seeds(seed_value=run+1)
      m = MLP2(widths=(300, 100), H=H, W=W, C=C)
      t = Trainer(m)
      t.save(run, 0, "SETOL/TEST")
  
  m = MLP2(widths=(300, 100), H=H, W=W, C=C)
  t = Trainer(m)
  for layer, opt, base_LR in hyper_params:
    if WHITEN:
      if layer == "all": continue
      layer = f"{layer}_WHITENED"
  
    for scale in range(SCALES):
      if LR_search:
        BS = 32
        LR = [lr * 2**scale for lr in base_LR]
        model_name = f"SETOL/{DS}/{opt}/{layer}/LR_{2**scale}"
      else:
        # BS search
        BS = 2 ** scale
        LR = base_LR
        model_name = f"SETOL/{DS}/{opt}/{layer}/BS_{BS}"

      for run in range(RUNS):
        if Trainer.metrics_path(run, model_name).exists(): continue
        print(model_name, run)
        
        starting_epoch = 0
        if RESTART:
          starting_epoch = last_epoch(run, model_name)
    
        loader = PreLoader(DS, TRAIN, TEST, batch_size=BS)
        loader.split_val(0.1)

        t.load(run, 0, "SETOL/TEST")  # Start with the same initial weights
        if WHITEN:
          whiten(m, LR)

        reset_random_seeds(seed_value=run+1)
        t.train_loop(model_name, run, 1000, loader, starting_epoch = starting_epoch,
          LR=LR, opt=opt,
          loss="CCE", early_stop=EarlyStopper(3, 0.0001))
        E = last_epoch(run, model_name)
        print(f"{model_name} Batch size {BS} converged at epoch {E}")
        print(Trainer.load_details(run, E, model_name))
        print()


if __name__ == "__main__":
  # NOTE: If you use this, then the non-determinism guarantees do not hold.
  RESTART = "RESTART" in sys.argv

  WHITEN = "WHITEN" in sys.argv
  
  LR_search = "LR" in sys.argv

  RUNS = 5  # number of times to run with a different seed.

  SCALES = 6  # Number of different learning rates or batch sizes to examine

  DS, C, H, W = "MNIST", 1, 28, 28
  if   "FASHION" in sys.argv: DS, C, H, W = "FASHION", 1, 28, 28
  elif "CIFAR10" in sys.argv: DS, C, H, W = "CIFAR10", 3, 32, 32

      
  main(DS, RUNS, LR_search, SCALES, WHITEN, C, H, W, RESTART)
