import sys
from pathlib import Path

import numpy as np
import torch

from trainer import Trainer, PreLoader, EarlyStopper
from models import MLP2
from pildataset import PILDataSet

from utils import last_epoch


def SVD_shuffle(c, xmin, SHUFFLE=True):
  # Decompose
  WM = c.weight
  if torch.cuda.is_available():
    with torch.no_grad():
      U, S, V = torch.linalg.svd(c.weight, full_matrices=False)
      S = S.cpu().numpy()
  else:
    U, S, V = np.linalg.svd(WM.cpu().numpy())
    U, V = map(lambda t: torch.nn.Parameter(torch.Tensor(t)), (U, V))

  # Shuffle
  if SHUFFLE:
    to_shuffle = np.where(S < np.sqrt(xmin))[0]
    S[to_shuffle] = S[to_shuffle[np.random.permutation(len(to_shuffle))]]

  # or Truncate
  else:
   S[np.where(S < np.sqrt(xmin))[0]] = 0

  # Recompose
  S = torch.Tensor(S).to(U.device)
  c.weight = torch.nn.Parameter(U @ torch.diag(S) @ V)


def shuffled_accuracy(t, loader, run, model_name, LR, device="cuda", SHUFFLE=True):
  print(f"{'shuffled' if SHUFFLE else 'truncated'} accuracy {model_name} run {run}")
  E = last_epoch(run, model_name)
  shuffled_train_acc  = np.zeros(E)
  shuffled_test_acc   = np.zeros(E)
  shuffled_train_loss = np.zeros(E)
  shuffled_test_loss  = np.zeros(E)

  for e in range(1, E+1):
    details = t.load_details(run, e, model_name)
    t.load(run, e, model_name)
    t.model.to(device)
    for c, lr, layer_id in zip(t.model.children(), LR, range(len(t.model.children()))):
      if lr > 0:
        xmin = details.loc[layer_id, "xmin"]
        SVD_shuffle(c, xmin, SHUFFLE)
    shuffled_train_acc[e-1], shuffled_train_loss[e-1] = t.evaluate(loader, "train")
    shuffled_test_acc [e-1], shuffled_test_loss [e-1] = t.evaluate(loader, "test")
    print('.', end="", flush=True)
    if e % 100 == 0: print()
  print()

  return shuffled_train_acc, shuffled_train_loss, shuffled_test_acc, shuffled_test_loss


def save_shuffled_accuracy(DS, OPT, layer, LR, runs, run_name, SHUFFLE):
  model_name = f"SETOL/{DS}/{OPT}/{layer}"

  for run in runs:
    save_file = f"./saved_models/{model_name}/{run_name(run)}_{'shuffled' if SHUFFLE else 'smoothed'}_accuracy.npy"
    if Path(save_file).exists(): continue

    shuffled_train_acc, shuffled_train_loss, shuffled_test_acc, shuffled_test_loss = shuffled_accuracy(t, loader, run, model_name, LR, SHUFFLE=SHUFFLE)
    with open(save_file, "wb") as fp:
      np.save(fp, shuffled_train_acc)
      np.save(fp, shuffled_train_loss)
      np.save(fp, shuffled_test_acc)
      np.save(fp, shuffled_test_loss)
    print(f"saved to {save_file}")


if __name__ == "__main__":
  ARGS = sys.argv.copy()
  DS, C, H, W = "MNIST", 1, 28, 28
  if   "FASHION" in ARGS: DS, C, H, W = "FASHION", 1, 28, 28
  elif "CIFAR10" in ARGS: DS, C, H, W = "CIFAR10", 3, 32, 32
      
  WHITEN = "WHITEN" in ARGS
  if WHITEN: 
    layer = f"{layer}_WHITENED"
      
  TRAIN = PILDataSet(True,  DS=DS)
  TEST  = PILDataSet(False, DS=DS)
  loader = PreLoader(DS, TRAIN, TEST, batch_size=10000)
  
  m = MLP2(widths=(300, 100), H=H, W=W, C=C)
  t = Trainer(m)

  layer_data = [
    ("layer0", [1, 0]),
    ("layer0_WHITENED", [1, 0]),
    ("layer1", [0, 1]),
    ("layer1_WHITENED", [0, 1]),
    ("all", [1, 1]),
  ]

  ALL_BS = [1,2,4,8,16,32]
  run_name = lambda r: f"BS = {ALL_BS[r]}"
  for DS in ["MNIST", "FASHION"]:
    for OPT in ["SGD", "ADAM"]:
      for layer, LR in layer_data:
        save_shuffled_accuracy(DS, OPT, layer, LR, range(6), run_name, SHUFFLE=False)
        save_shuffled_accuracy(DS, OPT, layer, LR, range(6), run_name, SHUFFLE=True)
