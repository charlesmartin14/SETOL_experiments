import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from weightwatcher import WeightWatcher

from trainer import Trainer
from utils import last_epoch

from utils import metric_error_bars, DF_error_bars

def set_styles():
  SMALL_SIZE = 10
  MEDIUM_SIZE = 16
  LARGE_SIZE = 30

  plt.rc('font', size=LARGE_SIZE)  # controls default text sizes
  plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
  plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
  plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
  plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
  plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
  plt.rc('figure', titlesize=LARGE_SIZE)  # fontsize of the figure title

def plot_loss(model_name, runs, run_name, trained_layers, WW_metric, TRAIN = True, LOSS = True):
  y_ax_name = f"{'train' if TRAIN else 'test'} {'loss' if LOSS else 'error'}"

  if not Trainer.save_dir(runs[0], 0, model_name).exists(): return

  fig, axes = plt.subplots(ncols=len(trained_layers), nrows=1, figsize=(6*len(trained_layers), 4))
  set_styles()

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



def plot_by_scales(DS, layer, scales, runs, WW_metrics, trained_layer = 0, search_param="BS"):
  blue_colors = plt.cm.Blues(np.linspace(0.5, 1, len(scales)))
  green_colors = plt.cm.Greens(np.linspace(0.5, 1, len(scales)))

  fig, axes = plt.subplots(nrows=1, ncols=len(WW_metrics)+1, figsize=(8*(1+len(WW_metrics)), 4))
  set_styles()

  means, stdevs = metric_error_bars(DS, layer, scales, runs, search_param=search_param)
  train_acc, train_loss, _, _, test_acc, test_loss = tuple(zip(*means))
  train_acc_SD, train_loss_SD, _, _, test_acc_SD, test_loss_SD = tuple(zip(*stdevs))

  mean_DFs, stdev_DFs = DF_error_bars(DS, layer, scales, runs, WW_metrics, search_param=search_param)

  def populate_tr(ax, scale, X, xerr):
    ax.errorbar(X, 1 - train_acc[scale], xerr=xerr, yerr=train_acc_SD[scale], fmt='.',
                color=blue_colors[scale], label=f"train error {search_param} = {2**scale}")

  def populate_te(ax, scale, X, xerr):
    ax.errorbar(X, 1 - test_acc[scale], xerr=xerr, yerr=test_acc_SD[scale], fmt='^',
                color=green_colors[scale], label=f"train error {search_param} = {2**scale}")


  layer_names = ["FC1", "FC2"]
  search_param_long = { "BS": "batch sizes", "LR": "learning rates"}[search_param]

  # For the first one just plot BS / LR directly.
  for scale in scales: populate_tr(axes[0], scale, X = 2**scale, xerr = 0)
  for scale in scales: populate_te(axes[0], scale, X = 2**scale, xerr = 0)

  box = axes[0].get_position()
  axes[0].set_position([box.x0, box.y0, box.width * 0.6, box.height])
  axes[0].set(title=f"MLP3: {search_param_long} vs. train/test error", xlabel=search_param_long)
  axes[0].legend(loc="center left", bbox_to_anchor=(1.1, 0.5))

  # Now fill the remaining plots with the desired WW_metrics.
  for ax, WW_metric in zip(axes[1:], WW_metrics):
    for scale, mean_details, stdev_details in zip(scales, mean_DFs, stdev_DFs):
      populate_tr( ax, scale,
        X     = mean_details.loc[trained_layer, WW_metric],
        xerr  = stdev_details.loc[trained_layer, WW_metric])
    for scale, mean_details, stdev_details in zip(scales, mean_DFs, stdev_DFs):
      populate_te( ax, scale,
        X = mean_details.loc[trained_layer, WW_metric],
        xerr = stdev_details.loc[trained_layer, WW_metric])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])


    ax.set(title=f"MLP3: {WW_metric} for {layer_names[trained_layer]} vs. train/test error\nVarious {search_param_long} considered", xlabel= WW_metric)
    ax.legend(loc="center left", bbox_to_anchor=(1.1, 0.5))


def plot_over_epochs(model_name, runs, run_name, WW_metric, layers):
  fig, axes = plt.subplots(nrows=1, ncols=len(layers), figsize = (6*len(layers), 4))
  set_styles()

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


def plot_shuffled_accuracy(DS, layer, LR, runs, run_name, SHUFFLE):
  fig, axes = plt.subplots(ncols = 4, nrows = 1, figsize=(18, 4))
  set_styles()

  SHUFFLE = 'shuffled' if SHUFFLE else 'smoothed'

  for run in runs:
    model_name = f"SETOL/{DS}/{layer}"
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

