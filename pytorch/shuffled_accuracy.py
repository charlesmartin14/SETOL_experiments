import sys
from pathlib import Path

import numpy as np
import torch

from trainer import Trainer, PreLoader, EarlyStopper
from models import MLP3
from pildataset import PILDataSet

from utils import last_epoch


def SVD_truncate(c, lambda_min):
  # Decompose
  WM = c.weight
  if torch.cuda.is_available():
    with torch.no_grad():
      U, S, V = torch.linalg.svd(c.weight, full_matrices=False)
      S = S.cpu().numpy()
  else:
    U, S, V = np.linalg.svd(WM.cpu().numpy())
    U, V = map(lambda t: torch.nn.Parameter(torch.Tensor(t)), (U, V))

  # Truncate
  S[np.where(S < np.sqrt(lambda_min))[0]] = 0

  # Recompose
  S = torch.Tensor(S).to(U.device)
  c.weight = torch.nn.Parameter(U @ torch.diag(S) @ V)


def truncated_accuracy(model_name, t, loader, LR, runs, device="cuda", XMIN=True):
  FIELD = 'xmin' if XMIN else "detX_val"
  FIELD_short = ['detX', 'xmin'][XMIN]

  for run in runs:
    save_file = f"./saved_models/{model_name}/{FIELD_short}_truncated_accuracy_run_{run}.npy"
    if Path(save_file).exists():
      print(f"Found path {save_file}")
      continue

    # Create a placeholder so that another process will not compete with this one.
    print(f"created a stub of {save_file}")
    with open(save_file, "wb") as fp:
      np.save(fp, np.zeros(1))

    print(f"truncated {FIELD_short} accuracy {model_name} run {run}")
    E = last_epoch(run, model_name)
    trunc_train_acc  = np.zeros(E+1)
    trunc_train_loss = np.zeros(E+1)
    trunc_val_acc    = np.zeros(E+1)
    trunc_val_loss   = np.zeros(E+1)
    trunc_test_acc   = np.zeros(E+1)
    trunc_test_loss  = np.zeros(E+1)

    details = t.load_details(run, model_name)
    for e in range(1, E+1):
      t.load(run, e, model_name)
      t.model.to(device)
      for c, lr, layer_id in zip(t.model.children(), LR, range(len(t.model.children()))):
        if lr > 0:
          lambda_min = details.query(f"epoch == {e}").loc[layer_id, FIELD]
          SVD_truncate(c, lambda_min)
      trunc_train_acc[e], trunc_train_loss[e] = t.evaluate(loader, "train")
      trunc_val_acc  [e], trunc_val_loss  [e] = t.evaluate(loader, "val")
      trunc_test_acc [e], trunc_test_loss [e] = t.evaluate(loader, "test")
      print('.', end="", flush=True)
      if e % 100 == 0: print()
    print()

    with open(save_file, "wb") as fp:
      np.save(fp, trunc_train_acc)
      np.save(fp, trunc_train_loss)
      np.save(fp, trunc_val_acc)
      np.save(fp, trunc_val_loss)
      np.save(fp, trunc_test_acc)
      np.save(fp, trunc_test_loss)
    print(f"saved to {save_file}")


def main(DS, RUNS, SCALES, search_param, C=1, H=28, W=28):
  TRAIN = PILDataSet(True,  DS=DS)
  TEST  = PILDataSet(False, DS=DS)
  loader = PreLoader(DS, TRAIN, TEST, batch_size=10000)
  loader.split_val(0.1)

  m = MLP3(widths=(300, 100), H=H, W=W, C=C)
  t = Trainer(m)

  layer_data = [
    ("all", [1, 1]),
    ("FC1", [1, 0]),
    ("FC2", [0, 1]),
  ]

  for layer, LR in layer_data:
    for scale in range(SCALES):
      model_name = f"SETOL/{DS}/{layer}/{search_param}_{2**scale}"
      truncated_accuracy(model_name, t, loader, LR, range(RUNS), XMIN=True)
      truncated_accuracy(model_name, t, loader, LR, range(RUNS), XMIN=False)


if __name__ == "__main__":
  ARGS = sys.argv.copy()

  DS, C, H, W = "MNIST", 1, 28, 28
  if   "FASHION" in ARGS: DS, C, H, W = "FASHION", 1, 28, 28
  elif "CIFAR10" in ARGS: DS, C, H, W = "CIFAR10", 3, 32, 32

  RUNS = 5
  SCALES = 6

  search_param = "BS"
  if "LR" in ARGS: search_param = "LR"

  main(DS, RUNS, SCALES, search_param, C, H, W)
