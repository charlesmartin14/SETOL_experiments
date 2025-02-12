import sys
from pathlib import Path

import numpy as np
import torch

from trainer import Trainer, PreLoader, EarlyStopper
from models import MLP3
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
    ("all", [0.01, 0.01, 0.01]),
    ("FC1", [0.01,    0,    0]),
    ("FC2", [   0, 0.01,    0]),
]


def main(DS, search_param, SCALES, RUNS, C=1, H=28, W=28, RESTART=False):
  assert search_param in ("BS", "LR"), search_param

  TRAIN = PILDataSet(True,  DS=DS)
  TEST  = PILDataSet(False, DS=DS)

  # Ensure that a different initialization exists for each seed.
  m = MLP3(widths=(300, 100), H=H, W=W, C=C)
  for run in range(RUNS):
    if not Trainer.save_dir(run, 0, "SETOL/TEST").exists():
      loader = PreLoader(DS, TRAIN, TEST, batch_size=10000)
      reset_random_seeds(seed_value=run+1)
      for c in m.children(): torch.nn.init.xavier_normal_(c.weight)
      t = Trainer(m)
      t.train_loop("SETOL/TEST", run, 1, loader)
      t.save(run, 0, "SETOL/TEST")
  
  t = Trainer(m)
  for layer, base_LR in hyper_params:
    for scale in range(SCALES):
      if search_param == "LR":
        BS = 32
        LR = [lr * 2**scale for lr in base_LR]
        model_name = f"SETOL/{DS}/{layer}/LR_{2**scale}"
      else:
        # BS search
        BS = 2 ** scale
        LR = base_LR
        model_name = f"SETOL/{DS}/{layer}/BS_{BS}"

      for run in range(RUNS):
        if Trainer.metrics_path(run, model_name).exists():
          print(f"Found path {Trainer.metrics_path(run, model_name)}")
          continue
        print(model_name, run)
        
        starting_epoch = 0
        if RESTART:
          starting_epoch = last_epoch(run, model_name)
    
        loader = PreLoader(DS, TRAIN, TEST, batch_size=BS)
        loader.split_val(0.1)

        t.load(run, 0, "SETOL/TEST")  # Start with the same initial weights

        reset_random_seeds(seed_value=run+1)
        t.train_loop(model_name, run, 1000, loader, starting_epoch = starting_epoch,
          LR=LR,
          loss="CCE", early_stop=EarlyStopper(3, 0.0001))
        E = last_epoch(run, model_name)
        print(f"{model_name} Batch size {BS} converged at epoch {E}")
        print(Trainer.load_details(run, model_name).query(f"epoch == {E}"))
        print()


if __name__ == "__main__":
  # NOTE: If you use this, then the non-determinism guarantees do not hold.
  RESTART = "RESTART" in sys.argv

  RUNS = 5  # number of times to run with a different seed.

  SCALES = 6  # Number of different learning rates or batch sizes to examine

  DS, C, H, W = "MNIST", 1, 28, 28
  if   "FASHION" in sys.argv: DS, C, H, W = "FASHION", 1, 28, 28
  elif "CIFAR10" in sys.argv: DS, C, H, W = "CIFAR10", 3, 32, 32

  search_param = "BS"
  if "LR" in sys.argv: search_param = "LR"

  main(DS, search_param, SCALES, RUNS, C, H, W, RESTART)
