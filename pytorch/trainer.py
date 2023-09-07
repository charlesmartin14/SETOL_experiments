from pathlib import Path
from time import time

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from weightwatcher import WeightWatcher

class EarlyStopper:
  def __init__(self, patience=1, min_delta=0):
    self.patience = patience
    self.min_delta = min_delta
    self.counter = 0
    self.min_loss = np.inf

  def __call__(self, loss):
    if loss < self.min_loss - self.min_delta:
      self.min_loss = loss
      self.counter = 0
    elif loss >= self.min_loss - self.min_delta:
      self.counter += 1
    return self.counter >= self.patience



class Trainer(object):
  def __init__(self, model):
    self.model = model

    # attributes that cannot be initialized until training has begun.
    self.train_loss, self.train_acc, self.val_loss, self.val_acc, self.test_acc, self.test_loss = [None] * 6
    self.details, self.watcher = None, None

  @staticmethod
  def save_dir(run, e, model_name):
    save_dir = Path(f"./saved_models/{model_name}/run_{run}_ep_{e}")
    return save_dir

  @staticmethod
  def metrics_path(run, model_name):
    return Path(f"./saved_models/{model_name}/metrics_run_{run}.npy")

  @staticmethod
  def details_path(run, model_name):
    return Path(f"./saved_models/{model_name}/WW_details_run_{run}")

  def ww_analyze(self, run, epoch, model_name):
    if not hasattr(self, "watcher") or self.watcher is None:
      self.watcher = WeightWatcher(model=self.model, log_level="ERROR")

    details = self.watcher.analyze(layers=[], min_evals=50, vectors=True, detX=True, plot=False)

    details['epoch'] = epoch
    details['run_number'] = run
    details['model_name'] = model_name

    assert (epoch > 0 or self.train_acc[epoch] > 0), epoch
    details['train_acc' ] = self.train_acc[epoch]
    details['train_loss'] = self.train_loss[epoch]
    details['val_acc'   ] = self.val_acc[epoch]
    details['val_loss'  ] = self.val_loss[epoch]
    details['test_acc'  ] = self.test_acc[epoch]
    details['test_loss' ] = self.test_loss[epoch]

    if self.details is None:  self.details = details
    else:                     self.details = self.details.append(details)


  def save(self, run, e, model_name):
    self.save_metrics(run, model_name)

    save_path = self.save_dir(run, e, model_name)
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(self.model.state_dict(), save_path / "model")

  def save_metrics(self, run, model_name):
    metrics_file = self.metrics_path(run, model_name)
    if not metrics_file.parent.exists():
      metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "wb") as fp:
      np.save(fp, self.train_acc)
      np.save(fp, self.train_loss)
      np.save(fp, self.val_acc)
      np.save(fp, self.val_loss)
      np.save(fp, self.test_acc)
      np.save(fp, self.test_loss)

  def save_details(self, run, model_name):
    save_path = self.details_path(run, model_name)
    save_path.mkdir(parents=True, exist_ok=True)
    try:
      self.details.to_pickle(save_path)
    except AttributeError:
      pass


  def load(self, run, e, model_name):
    save_path = self.save_dir(run, e, model_name)
    if not (save_path / "model").exists():
      msg = f"Attempting to load model {model_name} run {run} ep {e} that does not exist"
      raise RuntimeError(msg)

    self.model.load_state_dict(torch.load(save_path / "model"))
    self.details = self.load_details(run, model_name)

    metrics = self.load_metrics(run, model_name)
    self.train_acc, self.train_loss, self.val_acc, self.val_loss, self.test_acc, self.test_loss = metrics

  @staticmethod
  def load_details(run, model_name):
    return pd.read_pickle(Trainer.details_path(run, model_name))

  @staticmethod
  def load_metrics(run, model_name, VERBOSE=False):
    return Trainer._load_metrics(Trainer.metrics_path(run, model_name), VERBOSE=VERBOSE)

  @staticmethod
  def _load_metrics(save_file, VERBOSE=False):
    if not isinstance(save_file, Path): save_file = Path(save_file)
    if not save_file.exists():
      if VERBOSE:
        print(f"save file {save_file} not found")
      return None, None, None, None, None, None

    with open(str(save_file), "rb") as fp:
      train_acc  = np.load(fp)
      train_loss = np.load(fp)
      val_acc    = np.load(fp)
      val_loss   = np.load(fp)
      test_acc   = np.load(fp)
      test_loss  = np.load(fp)
    return train_acc, train_loss, val_acc, val_loss, test_acc, test_loss


  def train_loop(self, model_name, run, epochs, batch_loader,
    starting_epoch = 0,
    save_every=1,
    LR = None,
    WD = None,
    preprocessing = None,
    post_batch_callbacks=(),
    post_epoch_callbacks=(),
    opt = "SGD", loss = "MSE",
    early_stop = None,
    VERBOSE=True,
    loss_aggregation="mean",
  ):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if starting_epoch > 0:
      print(f"Restarting from {starting_epoch}")
      self.load(run, starting_epoch, model_name)
      def extend(a):
        if len(a) < epochs:
          new_a = np.zeros((epochs+1,))
          new_a[:len(a)] = a
          return new_a
        return a
      self.train_acc  = extend(self.train_acc)
      self.train_loss = extend(self.train_loss)
      self.val_acc  = extend(self.val_acc)
      self.val_loss = extend(self.val_loss)
      self.test_acc  = extend(self.test_acc)
      self.test_loss = extend(self.test_loss)
    else:
      self.train_acc  = np.zeros((epochs+1,))
      self.train_loss = np.zeros((epochs+1,))
      self.val_acc   = np.zeros((epochs+1,))
      self.val_loss  = np.zeros((epochs+1,))
      self.test_acc   = np.zeros((epochs+1,))
      self.test_loss  = np.zeros((epochs+1,))


    def make_param_groups(children, LR, WD):
      if isinstance(LR, (int, float)): LR = [LR] * len(children)
      assert len(LR) == len(children), (LR, len(children))
      assert len(WD) == len(children), (WD, len(children))
      return [{
          "params": c.parameters(),
          "lr": lr,
          "weight_decay": wd
        } for lr, wd, c in zip(LR, WD, children)
        if lr > 0
      ]

    opt = opt.upper()
    if opt == "SGD":
      if LR is None: LR = [0.01 for _ in self.model.children()]
      if WD is None: WD = [0] * len(self.model.children())
      opt = torch.optim.SGD(make_param_groups(self.model.children(), LR, WD), momentum=0)
    elif opt == "ADAMW":
      if LR is None: LR = [0.001 for _ in self.model.children()]
      if WD is None: WD = [0.01] * len(self.model.children())
      opt = torch.optim.AdamW(make_param_groups(self.model.children(), LR, WD))
    elif opt == "ADAM":
      if LR is None: LR = [0.001 for _ in self.model.children()]
      if WD is None: WD = [0] * len(self.model.children())
      opt = torch.optim.Adam(make_param_groups(self.model.children(), LR, WD))
    else: raise ValueError(f"Optimizer {opt} not recognized")

    loss = loss.upper()
    if loss == "MSE": loss_fn = torch.nn.MSELoss(reduction=loss_aggregation)
    elif loss == "CCE": loss_fn = torch.nn.CrossEntropyLoss(reduction=loss_aggregation)
    else: raise ValueError(f"loss is wrong type {loss}")

    if preprocessing is None:
      preprocessing = torch.nn.Identity()

    # First save and analyze the model as initialized
    self.watcher = WeightWatcher(model=self.model, log_level="ERROR")
    self.ww_analyze(run, 0, model_name)
    self.save(run, 0, model_name)


    def inner_loop(e):
      N_tr = 0 # = number of train examples seen
      for b, (features, labels) in enumerate(batch_loader.batches("train")):
        opt.zero_grad()
        preds = self.model(features)
        loss = loss_fn(preds, labels)
        loss.backward()
        opt.step()

        acc = sum(torch.argmax(p)==torch.argmax(l) for p, l in zip(preds, labels))

        N_tr += len(labels)
        self.train_loss[e] += loss.item()
        self.train_acc[e] += acc

        for cb in post_batch_callbacks: cb(self.model, b, e)

      self.train_loss[e] /= ( (b+1) if loss_aggregation == "mean" else N_tr )
      self.train_acc[e] /= N_tr


    def eval_loop(e, mode):
      with torch.no_grad():
        N_te, loss, acc = 0, 0., 0.
        for b, (features, labels) in enumerate(batch_loader.batches(mode)):
          N_te += len(labels)
          preds = self.model(features) #.to(device))
          acc  += sum(torch.argmax(p)==torch.argmax(l) for p, l in zip(preds, labels))
          loss += loss_fn(preds, labels).item()

      loss /= ( (b+1) if loss_aggregation == "mean" else N_te )
      acc /= N_te
      return loss, acc


    self.model.to(device)

    prev_time = time()
    for e in range(starting_epoch + 1, epochs+1):
      self.model.train()
      inner_loop(e)

      # Compute val and test accuracy
      self.model.eval()
      if "val" in batch_loader.preloaded:
        self. val_loss[e], self. val_acc[e] = eval_loop(e, "val")
      self.test_loss[e], self.test_acc[e] = eval_loop(e, "test")


      # Compute the new WW metrics
      self.ww_analyze(run, e, model_name)
      self.save_details(run, model_name)

      if e % save_every == 0:
        self.save(run, e, model_name)
        if VERBOSE:
          tr_l = self.train_loss[e]
          tr_a = self.train_acc[e]
          te_a = self.test_acc[e]

          t = time() - prev_time
          prev_time = time()
          print(f"{model_name} run {run} epoch {e} loss {tr_l:0.04f} train accuracy {tr_a:0.04f} test accuracy {te_a:0.04f} {t:0.02f} seconds"
          f"\t alpha 1 {self.details.alpha[0]:0.03f}\t alpha 2 {self.details.alpha[1]:0.03f}")

      for cb in post_epoch_callbacks: cb(self.model, e)

      if early_stop is not None:
        if early_stop(self.train_loss[e]): break


  def evaluate(self, loader, mode, device="cuda", loss_fn=None):
    if loss_fn is None: loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
      self.model.to(device)
      N = 0
      acc = 0
      loss = 0
      for features, labels in loader.batches(mode, shuffle=False, batch_size=10000, device=device):
        preds = self.model(features)
        loss += loss_fn(preds, labels).item()
        acc += sum(torch.argmax(p)==torch.argmax(l) for p, l in zip(preds, labels))
        N += len(labels)
    return acc / N, loss / N


class PreLoader(object):
  """Pre-loading the entire dataset can be a good thing for speeding up CIFAR, but on larger tasks it's infeasible"""
  @staticmethod
  def preload_data(train_DS, test_DS, ds_name, device="cuda"):
    save_file = Path(f"datasets/{ds_name}_data")
    with torch.no_grad():
      if save_file.exists(): preloaded = torch.load(save_file)
      else:
        def extract(DS):
          loader = DataLoader(DS, batch_size=len(DS), shuffle=False)
          images, labels = [item.to("cpu") for item in next(iter(loader))]
          return {
            "images" : images,
            "labels" : F.one_hot(labels).requires_grad_(False)
          }
        preloaded = {
          "train" : extract(train_DS),
          "test"  : extract(test_DS),
        }
        torch.save(preloaded, save_file)

    preloaded["train"]["images"] = preloaded["train"]["images"].to(device)
    preloaded["train"]["labels"] = preloaded["train"]["labels"].to(device).to(torch.float32)
    preloaded["test" ]["images"] = preloaded["test" ]["images"].to(device)
    preloaded["test" ]["labels"] = preloaded["test" ]["labels"].to(device).to(torch.float32)
    return preloaded

  def __init__(self, DS_name, TRAIN, TEST, batch_size=16, device = None):
    self.DS_name, self.batch_size = DS_name, batch_size
    assert isinstance(TRAIN, Dataset), type(TRAIN)
    assert isinstance(TEST, Dataset), type(TEST)

    if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
    self.device = device
    self.preloaded = self.preload_data(TRAIN, TEST, DS_name, device=self.device)


  def split_val(self, fraction):
    assert 0 < fraction < 1, fraction

    N = len(self.preloaded["train"]["labels"])
    inds = np.random.permutation(N)[:int(fraction * N)]
    self.preloaded["val"] = {}
    self.preloaded["val"]["images"] = self.preloaded["train"]["images"][inds]
    self.preloaded["val"]["labels"] = self.preloaded["train"]["labels"][inds]
    
    A = np.ones(N)
    A[inds] = -1
    inds = np.where(A>=0)[0]
    self.preloaded["train"]["images"] = self.preloaded["train"]["images"][inds]
    self.preloaded["train"]["labels"] = self.preloaded["train"]["labels"][inds]
    

  def batches(self, mode, shuffle=True, batch_size=None, device="cuda"):
    assert mode in ("train", "test", "val"), mode
    if batch_size is None: batch_size = self.batch_size

    N = len(self.preloaded[mode]["images"])
    if shuffle: shuffled = torch.randperm(N, device=self.device)
    else:       shuffled = torch.  arange(N, device=self.device, dtype=torch.int32)

    self.preloaded[mode]["images"].to(device)
    self.preloaded[mode]["labels"].to(device)
    for i in range(int(np.ceil(N / batch_size))):
      yield self.preloaded[mode]["images"].index_select(0, shuffled[i*batch_size:(i+1)*batch_size]), \
        self.preloaded[mode]["labels"].index_select(0, shuffled[i*batch_size:(i+1)*batch_size])

