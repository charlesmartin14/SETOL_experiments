import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from weightwatcher import WeightWatcher

from trainer import Trainer
from utils import last_epoch


def plot_loss(model_name, runs, run_name, trained_layers, WW_metric, TRAIN = True, LOSS = True):
  y_ax_name = f"{'train' if TRAIN else 'test'} {'loss' if LOSS else 'error'}"

  if not Trainer._save_dir(runs[0], 0, model_name).exists(): return

  fig, axes = plt.subplots(ncols=len(trained_layers), nrows=1, figsize=(6*len(trained_layers), 4))

  if len(trained_layers) == 1: axes = [axes]

  SKIP = 2
  for ax, layer in zip(axes, trained_layers):
    for run in runs:
      train_acc, train_loss, test_acc, test_loss = Trainer.load_metrics(run, model_name)
      last_epoch = np.argmin(train_acc[1:] > 0) or len(train_acc)-1
      X = np.arange(SKIP, last_epoch(run, model_name))

      metric_vals = np.zeros(X.shape)
      for e in X:
        details = Trainer.load_details(run, e, model_name)
        metric_vals[e-SKIP] = details.loc[layer, WW_metric]
      if LOSS:
        if TRAIN: ax.plot(metric_vals, train_loss[X], '+', label=run_name(run))
        else:     ax.plot(metric_vals, test_loss[X], '+', label=run_name(run))
      else:
        if TRAIN: ax.plot(metric_vals, 1-train_acc[X], '+', label=run_name(run))
        else:     ax.plot(metric_vals, 1-test_acc[X], '+', label=run_name(run))
    ax.legend()
    ax.set(title=f"{model_name}\nAll epochs\n vs. {y_ax_name} {WW_metric} for layer {layer}",
         ylabel=y_ax_name, xlabel=WW_metric, ylim=(0, None))


def plot_over_epochs(model_name, runs, run_name, WW_metric, layers):
  fig, axes = plt.subplots(nrows=1, ncols=len(layers), figsize = (6*len(layers), 4))

  for run in runs:
    E = last_epoch(run, model_name)
    metric_data = np.zeros((len(layers), E))
    for e in range(E):
      metric_data[:, e] = Trainer.load_details(run, e+1, model_name).loc[layers, WW_metric]
    for l, ax in zip(layers, axes):
      ax.plot(metric_data[l,:], '+', label = run_name(run))
  for l, ax in zip(layers, axes):
    ax.set(title=f"{model_name}\nlayer {l}", xlabel="epoch", ylabel=WW_metric)
    ax.legend()


def plot_shuffled_accuracy(DS, OPT, layer, LR, runs, run_name, SHUFFLE):
  fig, axes = plt.subplots(ncols = 4, nrows = 1, figsize=(16, 4))

  SHUFFLE = 'shuffled' if SHUFFLE else 'smoothed'

  for run in runs:
    train_acc, train_loss, test_acc, test_loss = Trainer.load_metrics(run, model_name)

    save_file = f"./saved_models/{model_name}/{run_name(run)}_{SHUFFLE}_accuracy.npy"
    with open(save_file, "rb") as fp:
      shuffled_train_acc  = np.load(fp)
      shuffled_train_loss = np.load(fp)
      shuffled_test_acc   = np.load(fp)
      shuffled_test_loss  = np.load(fp)

    axes[0].plot(train_acc  - shuffled_train_acc , '+', label = run_name(run))
    axes[1].plot(train_loss - shuffled_train_loss,     '+', label = run_name(run))
    axes[2].plot(test_acc   - shuffled_test_acc  , '+', label = run_name(run))
    axes[3].plot(test_loss  - shuffled_test_loss,      '+', label = run_name(run))

  axes[0].set(xlabel="epochs", ylabel=f"{SHUFFLE} train error", title=f"{model_name}\nDelta between train error and {SHUFFLE} train error")
  axes[1].set(xlabel="epochs", ylabel=f"{SHUFFLE} train loss",  title=f"{model_name}\nDelta between train loss and {SHUFFLE} train loss")
  axes[2].set(xlabel="epochs", ylabel=f"{SHUFFLE} test error",  title=f"{model_name}\nDelta between test error and {SHUFFLE} test error")
  axes[3].set(xlabel="epochs", ylabel=f"{SHUFFLE} test loss",   title=f"{model_name}\nDelta between test loss and {SHUFFLE} test loss")

  for ax in axes: ax.legend()

