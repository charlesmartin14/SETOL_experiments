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
      train_acc, train_loss, _, _, test_acc, test_loss = Trainer.load_metrics(run, model_name)
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


from utils import metric_error_bars, DF_error_bars

def plot_by_scales(DS, OPT, layer, scales, runs, WW_metrics, trained_layer = 0, search_param="BS"):
  blue_colors = plt.cm.Blues(np.linspace(0.5, 1, len(scales)))
  green_colors = plt.cm.Greens(np.linspace(0.5, 1, len(scales)))

  fig, axes = plt.subplots(nrows=1, ncols=len(WW_metrics), figsize=(8*len(WW_metrics), 4))

  means, stdevs = metric_error_bars(DS, OPT, layer, scales, runs, search_param=search_param)
  train_acc, train_loss, _, _, test_acc, test_loss = tuple(zip(*means))
  train_acc_SD, train_loss_SD, _, _, test_acc_SD, test_loss_SD = tuple(zip(*stdevs))

  mean_DFs, stdev_DFs = DF_error_bars(DS, OPT, layer, scales, runs, WW_metrics, search_param=search_param)

  for ax, WW_metric in zip(axes, WW_metrics):
    for scale, mean_details, stdev_details in zip(scales, mean_DFs, stdev_DFs):
      X = mean_details.loc[trained_layer, WW_metric]
      xerr = stdev_details.loc[trained_layer, WW_metric]
      Y = 1 - train_acc[scale]
      yerr = train_acc_SD[scale]
      ax.errorbar(X, Y, xerr=xerr, yerr=yerr, fmt='.', color=blue_colors[scale], label=f"train error {search_param} = {2**scale}")
    for scale, mean_details, stdev_details in zip(scales, mean_DFs, stdev_DFs):
      X = mean_details.loc[trained_layer, WW_metric]
      xerr = stdev_details.loc[trained_layer, WW_metric]
      Y = 1 - test_acc[scale]
      yerr = test_acc_SD[scale]
      ax.errorbar(X, Y, xerr=xerr, yerr=yerr, fmt='^', color=green_colors[scale], label=f"test error {search_param} = {2**scale}")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])

    model_base = f"{DS} {OPT} {layer} {search_param} search"
    ax.set(title=f"{model_base}\n{WW_metric}\n for layer {trained_layer}", xlabel= WW_metric)
    ax.legend(loc="center left", bbox_to_anchor=(1.1, 0.5))


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
  fig, axes = plt.subplots(ncols = 4, nrows = 1, figsize=(18, 4))

  SHUFFLE = 'shuffled' if SHUFFLE else 'smoothed'

  for run in runs:
    model_name = f"SETOL/{DS}/{OPT}/{layer}"
    train_acc, train_loss, _, _, test_acc, test_loss = Trainer.load_metrics(run, model_name)
    if train_acc is None:
      print(f"metrics for {model_name} not found")
      continue

    save_file = f"./saved_models/{model_name}/{run_name(run)}_{SHUFFLE}_accuracy.npy"
    with open(save_file, "rb") as fp:
      shuffled_train_acc  = np.load(fp)
      shuffled_train_loss = np.load(fp)
      shuffled_test_acc   = np.load(fp)
      shuffled_test_loss  = np.load(fp)

    E = last_epoch(run, model_name)
    axes[0].plot(train_acc[1:E]  - shuffled_train_acc[1:] , '-', label = run_name(run))
    axes[1].plot(train_loss[1:E] - shuffled_train_loss[1:], '-', label = run_name(run))
    axes[2].plot(test_acc[1:E]   - shuffled_test_acc[1:]  , '-', label = run_name(run))
    axes[3].plot(test_loss[1:E]  - shuffled_test_loss[1:],  '-', label = run_name(run))

  axes[0].set(xlabel="epochs", ylabel=f"{SHUFFLE} train error", title=f"{model_name}\ntrain error - {SHUFFLE} train error")
  axes[1].set(xlabel="epochs", ylabel=f"{SHUFFLE} train loss",  title=f"{model_name}\ntrain loss - {SHUFFLE} train loss")
  axes[2].set(xlabel="epochs", ylabel=f"{SHUFFLE} test error",  title=f"{model_name}\ntest error - {SHUFFLE} test error")
  axes[3].set(xlabel="epochs", ylabel=f"{SHUFFLE} test loss",   title=f"{model_name}\ntest loss - {SHUFFLE} test loss")

  for ax in axes: ax.legend()

