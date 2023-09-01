import sys
from pathlib import Path

import numpy as np
import torch

from trainer import Trainer, PreLoader, EarlyStopper
from models import MLP2
from pildataset import PILDataSet
from utils import last_epoch


DETERMINISTIC = True

if DETERMINISTIC:
  import random
  import os

  os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

  torch.use_deterministic_algorithms(True)
  def reset_random_seeds(seed_value=1):
    os.environ['PYTHONHASHSEED']=str(seed_value)
    torch.manual_seed(0)
    np.random.seed(seed_value)
    random.seed(seed_value)

  reset_random_seeds()
else:
  reset_random_seeds = lambda: None



hyper_params = [
    ("all",    "SGD", [0.01, 0.01, 0.01]),
    ("FC1", "SGD", [0.01,    0,    0]),
    ("FC2", "SGD", [   0, 0.01,    0]),
    
    ("all",    "ADAM", [0.001, 0.001, 0.001]),
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


if __name__ == "__main__":
  ARGS = sys.argv.copy()
  
  DS, C, H, W = "MNIST", 1, 28, 28
  if   "FASHION" in ARGS: DS, C, H, W = "FASHION", 1, 28, 28
  elif "CIFAR10" in ARGS: DS, C, H, W = "CIFAR10", 3, 32, 32
  
  
  WHITEN = "WHITEN" in ARGS
      
  
  # NOTE: If you use this, then the non-determinism guarantees do not hold.
  RESTART = "RESTART" in ARGS
  
  TRAIN = PILDataSet(True,  DS=DS)
  TEST  = PILDataSet(False, DS=DS)
  ALL_BS = [1,2,4,8,16,32]
  
  
  m = MLP2(widths=(300, 100), H=H, W=W, C=C)
  t = Trainer(m)
  for layer, op, LR in hyper_params:
    if WHITEN and layer != "all":
      layer = f"{layer}_WHITENED"
  
    for run, BS in enumerate(ALL_BS):
      model_name = f"SETOL/{DS}/{opt}/{layer}"
      print(model_name, run)
      
      starting_epoch = 0
      if RESTART:
        starting_epoch = last_epoch(run, model_name)
  
      loader = PreLoader(DS, TRAIN, TEST, batch_size=BS)
  
      t.load(0, 0, f"SETOL/MNIST/SGD/FC1")  # Start with the same initial weights
      if WHITEN:
        whiten(m, LR)
  
      reset_random_seeds()
      t.train_loop(model_name, run, 1000, loader, starting_epoch = starting_epoch,
        LR=LR, opt=opt,
        loss="CCE", early_stop=EarlyStopper(3, 0.0001))
      
      print(f"{model_name} Batch size {BS} converged at epoch {last_epoch(model_name, run)}")
      print(t.details)
      print()
