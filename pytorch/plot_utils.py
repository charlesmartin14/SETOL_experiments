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


def plot_loss(DS, layer, search_param, scale, runs, plot_layers, WW_metric, LOSS = True):
  model_name = f"SETOL/{DS}/{layer}/{search_param}_{2**scale}"

  runs = [
    run for run in runs
    if Trainer.save_dir(run, 0, model_name).exists()
  ]
  if not runs: return

  L = len(plot_layers)
  fig, axes = plt.subplots(ncols=2, nrows=L, figsize=(12, 4*len(plot_layers)))
  set_styles()

  SKIP = 2
  if len(plot_layers) == 1: axes = [axes]
  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,
                    wspace=0.4,
                    hspace=0.5)
  for ax_row, layer in zip(axes, plot_layers):
    for ax, TRAIN in zip(ax_row, [True, False]):
      y_ax_name = f"{'TRAIN' if TRAIN else 'TEST'} {'loss' if LOSS else 'error'}"
      for run in runs:
        train_acc, train_loss, _, _, test_acc, test_loss = Trainer.load_metrics(run, model_name)
        X = np.arange(SKIP, last_epoch(run, model_name))
  
        metric_vals = np.zeros(X.shape)
        details = Trainer.load_details(run, model_name)
        for e in X:
          metric_vals[e-SKIP] = details.query(f"epoch == {e}").loc[layer, WW_metric]
        if LOSS:
          if TRAIN: ax.plot(metric_vals, train_loss[X], '+', label=f"seed={run+1}", alpha=0.5)
          else:     ax.plot(metric_vals, test_loss [X], '+', label=f"seed={run+1}", alpha=0.5)
        else:
          if TRAIN: ax.plot(metric_vals, 1-train_acc[X], '+', label=f"seed={run+1}", alpha=0.5)
          else:     ax.plot(metric_vals, 1-test_acc [X], '+', label=f"seed={run+1}", alpha=0.5)
      ax.legend()
      ax.set(title=f"{model_name}\n{y_ax_name} vs. {WW_metric} for layer {layer}",
           ylabel=y_ax_name, xlabel=WW_metric, ylim=(0, None))



def plot_by_scales(DS, layer, scales, runs, WW_metrics, plot_layer = 0, search_param="BS"):
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
        X     = mean_details.loc[plot_layer, WW_metric],
        xerr  = stdev_details.loc[plot_layer, WW_metric])
    for scale, mean_details, stdev_details in zip(scales, mean_DFs, stdev_DFs):
      populate_te( ax, scale,
        X = mean_details.loc[plot_layer, WW_metric],
        xerr = stdev_details.loc[plot_layer, WW_metric])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])


    ax.set(title=f"MLP3: {WW_metric} for {layer_names[plot_layer]} vs. train/test error\nVarious {search_param_long} considered", xlabel= WW_metric)
    ax.legend(loc="center left", bbox_to_anchor=(1.1, 0.5))


def plot_over_epochs(DS, layer, search_param, scale, runs, WW_metric, layers):
  model_name = f"SETOL/{DS}/{layer}/{search_param}_{2**scale}"


  fig, axes = plt.subplots(nrows=1, ncols=len(layers), figsize = (6*len(layers), 4))
  set_styles()

  if len(layers) == 1: axes = [axes]

  for run in runs:
    details = Trainer.load_details(run, model_name)
    E = last_epoch(run, model_name)
    for l, ax in zip(layers, axes):
      metric_data = np.zeros((E,))
      for e in range(E):
        metric_data[e] = details.query(f'epoch == {e+1}').loc[l, WW_metric]
      ax.plot(metric_data, '+', label = f"seed = {run}")
  for l, ax in zip(layers, axes):
    ax.set(title=f"{model_name}\nlayer FC{l+1}", xlabel="epoch", ylabel=WW_metric)
    ax.legend()



def plot_truncated_accuracy_over_epochs(DS, layer, search_param, scale, runs,
    XMIN=True,      # Whether to use xmin, or detX as the tail demarcation
    ylims=None,     # Maximum values to set for each plot so they can be compared
    E0=4  # Skip the first few epochs so that the variance is contained.
  ):
  fig, axes = plt.subplots(ncols = 2, nrows = 1, figsize=(13, 4))
  set_styles()

  if ylims is None: ylims = (None, None)

  FIELD = 'xmin' if XMIN else "detX_val_unrescaled"
  FIELD_short = ['detX', 'xmin'][XMIN]

  model_name = f"SETOL/{DS}/{layer}/{search_param}_{2**scale}"
  Emin = min(last_epoch(run, model_name) for run in runs)

  train_acc        = np.zeros((len(runs), Emin+1))
  train_loss       = np.zeros((len(runs), Emin+1))
  test_acc         = np.zeros((len(runs), Emin+1))
  test_loss        = np.zeros((len(runs), Emin+1))

  trunc_train_acc  = np.zeros((len(runs), Emin+1))
  trunc_train_loss = np.zeros((len(runs), Emin+1))
  trunc_test_acc   = np.zeros((len(runs), Emin+1))
  trunc_test_loss  = np.zeros((len(runs), Emin+1))


  for run in runs:
    metrics = Trainer.load_metrics(run, model_name)
    if metrics[0] is None:
      print(f"metrics for {model_name} not found")
      continue

    train_acc[run,:Emin+1], train_loss[run,:Emin+1], _, _, test_acc[run,:Emin+1], test_loss[run,:Emin+1] = [
      m[:Emin+1]
      for m in metrics
    ]


    # FIXME: Using old name smoothed rather than truncated until code finishes running.
    metrics = Trainer.load_metrics(run, model_name, save_file = f"{FIELD_short}_smoothed_accuracy_run_{run}.npy")
    if metrics[0] is None:
      print(f"truncated metrics for {model_name} not found")
      continue

    trunc_train_acc[run,:Emin+1], trunc_train_loss[run,:Emin+1], _, _, trunc_test_acc[run,:Emin+1], trunc_test_loss[run,:Emin+1] = [
      m[:Emin+1]
      for m in metrics
    ]
    

  
  plot_one = lambda ax, Y, label: ax.errorbar(np.arange(E0, Emin+1), np.mean(Y[:,E0:], axis=0),
    yerr=np.std(Y[:,E0:], axis=0), fmt='-', label=r'$\Delta$ ' + label)

  plot_one(axes[0], train_acc           - trunc_train_acc, "train error")
  plot_one(axes[0], test_acc            - trunc_test_acc , "test error")
  plot_one(axes[1], trunc_train_loss - train_loss        , "train loss")
  plot_one(axes[1], trunc_test_loss  - test_loss         , "test loss")


  common_title = f"{search_param} = {2**scale} trained layer(s): {layer}"
  axes[0].set(xlabel="epochs", ylabel=r"$\Delta$ error", title=f"{common_title}\ntrain error - truncated train error")
  axes[1].set(xlabel="epochs", ylabel=r"$\Delta$ loss",  title=f"{common_title}\ntruncated train loss - train loss")


  axes[0].set(ylim=(-0.0075, ylims[0]))
  axes[1].set(ylim=(-0.02, ylims[1]))
  for ax in axes:
    ax.legend()
    ax.axhline(0, color="gray", zorder=-1)

