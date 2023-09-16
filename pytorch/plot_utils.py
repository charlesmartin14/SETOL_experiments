from pathlib import Path

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


def plot_loss(DS, layer, search_param, scale, runs, plot_layers, WW_metric, LOSS = True, ylim=None):
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

      if ylim is None: ylim = (0, None)
      ax.set(title=f"{model_name}\n{y_ax_name} vs. {WW_metric} for layer {layer}",
           ylabel=y_ax_name, xlabel=WW_metric, ylim=ylim)



def plot_by_scales(DS, layer, scales, runs, WW_metrics,
  plot_layer = 0,
  search_param="BS",
  save_dir = None,
):
  blue_colors = plt.cm.Blues(np.linspace(0.5, 1, len(scales)))
  green_colors = plt.cm.Greens(np.linspace(0.5, 1, len(scales)))
  red_colors = plt.cm.Reds(np.linspace(0.5, 1, len(scales)))
  if search_param == "BS": blue_colors[ 0] = green_colors[ 0] = red_colors[0]
  if search_param == "LR": blue_colors[-1] = green_colors[-1] = red_colors[0]

  fig, axes = plt.subplots(nrows=1, ncols=len(WW_metrics)+1, figsize=(8*(1+len(WW_metrics)) + 2, 4))
  set_styles()

  means, stdevs = metric_error_bars(DS, layer, scales, runs, search_param=search_param)
  train_acc, train_loss, _, _, test_acc, test_loss = tuple(zip(*means))
  train_acc_SD, train_loss_SD, _, _, test_acc_SD, test_loss_SD = tuple(zip(*stdevs))

  mean_DFs, stdev_DFs = DF_error_bars(DS, layer, scales, runs, WW_metrics, search_param=search_param)

  def populate_tr(ax, scale, X, xerr):
    ax.errorbar(
      X, 1 - train_acc[scale], xerr=xerr, yerr=train_acc_SD[scale], fmt='.', color=blue_colors[scale]
    )
    return f"{search_param} = {2**scale}"

  def populate_te(ax, scale, X, xerr):
    ax.errorbar(
      X, 1 - test_acc[scale], xerr=xerr, yerr=test_acc_SD[scale], fmt='^', color=green_colors[scale]
    )
    return f"{search_param} = {2**scale}"

  S = len(scales)
  def plot_legends(ax, tr_labels, te_labels):
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])

    tr_legend = ax.legend(handles=ax.containers[:S], labels=tr_labels, loc="center left", bbox_to_anchor=(1.0, 0.25))
    te_legend = ax.legend(handles=ax.containers[S:], labels=te_labels, loc="center left", bbox_to_anchor=(1.0, 0.75))

    tr_legend.set_title("Train set errors")
    tr_legend.get_title().set_fontsize(11)

    te_legend.set_title("Test set errors ")
    te_legend.get_title().set_fontsize(11)

    ax.add_artist(tr_legend)
    ax.add_artist(te_legend)
  

  layer_names = ["FC1", "FC2"]
  search_param_long = { "BS": "batch size", "LR": "learning rate"}[search_param]

  # For the first one just plot BS / LR directly.
  tr_labels = [ populate_tr(axes[0], scale, 2**scale, 0) for scale in scales ]
  te_labels = [ populate_te(axes[0], scale, 2**scale, 0) for scale in scales ]
  axes[0].set(title=f"MLP3: {search_param_long} vs. train/test error", xlabel=search_param_long, ylabel="error")

  plot_legends(axes[0], tr_labels, te_labels)
  # Now fill the remaining plots with the desired WW_metrics.
  layer_name = layer_names[plot_layer]
  for ax, WW_metric in zip(axes[1:], WW_metrics):
    tr_labels = [
      populate_tr( ax, scale,
        X     =  mean_details.loc[plot_layer, WW_metric],
        xerr  = stdev_details.loc[plot_layer, WW_metric])
        for scale, mean_details, stdev_details in zip(scales, mean_DFs, stdev_DFs)
    ]
    te_labels = [
      populate_te( ax, scale,
        X    =  mean_details.loc[plot_layer, WW_metric],
        xerr = stdev_details.loc[plot_layer, WW_metric])
      for scale, mean_details, stdev_details in zip(scales, mean_DFs, stdev_DFs)
    ]

    ax.set(title=f"MLP3: {WW_metric} for {layer_name} vs. train/test error\nVarious {search_param_long} considered",
      xlabel= WW_metric, ylabel="error")

    plot_legends(ax, tr_labels, te_labels)

  plt.tight_layout(rect=[0, 0, 0.6, 1.4]) 
  plt.subplots_adjust(wspace=0.80, hspace=0.95)
  
  if save_dir is not None:
    if not isinstance(save_dir, Path): save_dir = Path(save_dir)
    if not save_dir.exists(): save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"mlp3_quality_by_{search_param}_{layer}_{layer_name}.png", bbox_inches='tight')



def plot_over_epochs(DS, layer, search_param, scale, runs, WW_metric, layers):
  """ Plots a particular WW metric, such as "alpha", over the epochs of a series of training runs. Each trained layer is shown in a separate column
      DS, layer, search_param, scale: Fields that identify which experiment was done.
      WW_metric: A WeightWatcher metric to be plotted. 
      layers: A list of indices of layers for which WW_metric should be plotted. Valid indices are {0, 1}
  """
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
      ax.plot(metric_data, '+', label = f"seed = {run+1}")

  for l, ax in zip(layers, axes):
    ax.set(title=f"{model_name}\nlayer FC{l+1}", xlabel="epoch", ylabel=WW_metric)
    ax.legend()



def populate_metrics_all_epochs(DS, layer, search_param, scale, runs, TRUNC_field=None):
  model_name = f"SETOL/{DS}/{layer}/{search_param}_{2**scale}"

  Emin = min(last_epoch(run, f"SETOL/{DS}/{layer}/{search_param}_{2**scale}") for run in runs)
  train_acc        = np.zeros((len(runs), Emin+1))
  train_loss       = np.zeros((len(runs), Emin+1))
  test_acc         = np.zeros((len(runs), Emin+1))
  test_loss        = np.zeros((len(runs), Emin+1))


  save_file=None
  for run in runs:
    if TRUNC_field is not None:
      # FIXME: Using old name smoothed rather than truncated until code finishes running.
      save_file = f"{TRUNC_field}_smoothed_accuracy_run_{run}.npy"
    metrics = Trainer.load_metrics(run, model_name, save_file=save_file)
    if metrics[0] is None:
      print(f"metrics for {model_name} {save_file} not found")
      continue

    train_acc[run,:Emin+1], train_loss[run,:Emin+1], _, _, test_acc[run,:Emin+1], test_loss[run,:Emin+1] = [
      m[:Emin+1]
      for m in metrics
    ]

  return train_acc, train_loss, test_acc, test_loss


def populate_metrics_last_E(DS, layer, search_param, scales, runs, TRUNC_field=None, FLAT=False):
  train_acc   = np.zeros((len(scales), len(runs)))
  train_loss  = np.zeros((len(scales), len(runs)))
  test_acc    = np.zeros((len(scales), len(runs)))
  test_loss   = np.zeros((len(scales), len(runs)))

  save_file = None
  for scale_i, scale in enumerate(scales):
    model_name = f"SETOL/{DS}/{layer}/{search_param}_{2**scale}"
    for run_i, run in enumerate(runs):
      if TRUNC_field is not None:
        # FIXME: Using old name smoothed rather than truncated until code finishes running.
        save_file = f"{TRUNC_field}_smoothed_accuracy_run_{run}.npy"

      E = last_epoch(run, model_name)
      ind = (scale_i, run_i)

      metrics = Trainer.load_metrics(run, model_name, save_file=save_file)
      if metrics[0] is None: continue
      train_acc[ind], train_loss[ind], _, _, test_acc[ind], test_loss[ind] = [m[E] for m in metrics]

  if FLAT:
    train_acc  = train_acc.reshape((-1))
    train_loss = train_loss.reshape((-1))
    test_acc   = test_acc.reshape((-1))
    test_loss  = test_loss.reshape((-1))

  return train_acc, train_loss, test_acc, test_loss



def populate_WW_metric_all_epochs(DS, layer, search_param, scale, runs, Emin, WW_metric):
  model_name = f"SETOL/{DS}/{layer}/{search_param}_{2**scale}"

  WW_data = np.zeros((len(runs), Emin+1, 2))

  for run_i, run in enumerate(runs):
    details = Trainer.load_details(run, model_name)
    for epoch in range(1, Emin+1):
      WW_data[run_i, epoch, :] = details.query(f"epoch == {epoch}").loc[:, WW_metric]
  
  return WW_data



def populate_WW_metric_last_epoch(DS, layer, search_param, scales, runs, WW_metric):
  WW_data = np.zeros((len(scales), len(runs), 2))

  for scale_i, scale in enumerate(scales):
    model_name = f"SETOL/{DS}/{layer}/{search_param}_{2**scale}"
    for run_i, run in enumerate(runs):
      E = last_epoch(run, model_name)

      WW_data[scale_i, run_i, :] = Trainer.load_details(
        run, model_name).query(f"epoch == {E}").loc[:, WW_metric]
  
  return WW_data



def plot_truncated_accuracy_over_epochs(DS, layer, search_param, scale, runs,
    XMIN=True,      # Whether to use xmin, or detX as the tail demarcation
    ylims=None,     # Maximum values to set for each plot so they can be compared
    E0=4, # Skip the first few epochs so that the variance is contained.
    save_dir=None,  # place to save the images
  ):
  fig, axes = plt.subplots(ncols = 3, nrows = 1, figsize=(20, 4))
  set_styles()

  if ylims is None: ylims = (None, None)

  FIELD = 'xmin' if XMIN else "detX_val_unrescaled"
  FIELD_short = ['detX', 'xmin'][XMIN]

  train_acc, train_loss, test_acc, test_loss = populate_metrics_all_epochs(
    DS, layer, search_param, scale, runs, TRUNC_field=None)

  trunc_train_acc, trunc_train_loss, trunc_test_acc, trunc_test_loss = populate_metrics_all_epochs(
    DS, layer, search_param, scale, runs, TRUNC_field=FIELD_short)
  
  Emin = train_acc.shape[1]-1

  alpha = populate_WW_metric_all_epochs(DS, layer, search_param, scale, runs, Emin, "alpha")

  X = np.arange(E0, Emin+1)
  plot_one = lambda ax, Y, label, color=None: ax.errorbar(X, np.mean(Y[:,E0:], axis=0),
    yerr=np.std(Y[:,E0:], axis=0), fmt='-', label=r'$\Delta$ ' + label, color=color)

  # Error = 1 - accuracy
  plot_one(axes[0], train_acc        - trunc_train_acc, "train error")
  plot_one(axes[0], test_acc         - trunc_test_acc , "test error")
  plot_one(axes[1], trunc_train_loss - train_loss     , "train loss")
  plot_one(axes[1], trunc_test_loss  - test_loss      , "test loss")

  if layer in ("all", "FC1", "FC1_WHITENED"): plot_one(axes[2], alpha[:,:,0], label=r"FC1 $\alpha$", color="red")
  if layer in ("all", "FC2", "FC2_WHITENED"): plot_one(axes[2], alpha[:,:,1], label=r"FC2 $\alpha$", color="green")


  common_title = f"{search_param} = {2**scale} trained layer(s): {layer}"
  axes[0].set(xlabel="epochs", ylabel=r"$\Delta$ error", title=f"{common_title}\ntruncated train error - train error")
  axes[1].set(xlabel="epochs", ylabel=r"$\Delta$ loss",  title=f"{common_title}\ntruncated train loss - train loss")
  axes[2].set(xlabel="epochs", ylabel=r"$\alpha$",       title=f"{common_title}\n" + r"$\alpha$ for trained layers")

  axes[2].axhline(2, color="gray", zorder=-1)

  axes[0].set(ylim=(-0.0075, ylims[0]))
  axes[1].set(ylim=(-0.02, ylims[1]))
  axes[2].set(ylim=(1, None))
  for ax in axes:
    ax.legend()
    ax.axhline(0, color="gray", zorder=-1)

  if save_dir is not None:
    if not isinstance(save_dir, Path): save_dir = Path(save_dir)
    if not save_dir.exists(): save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"mlp3_trunc_error_by_epochs_{search_param}_{scale}_{layer}_{FIELD_short}.png", bbox_inches='tight')


def plot_truncated_errors_by_scales(DS, layer, search_param, scales, run,
    XMIN=True,      # Whether to use xmin, or detX as the tail demarcation
    save_dir = None,
  ):
  FIELD = 'xmin' if XMIN else "detX_val_unrescaled"
  FIELD_short = ['detX', 'xmin'][XMIN]

  fig, axes = plt.subplots(ncols = 3, nrows = 1, figsize=(20, 2))
  set_styles()

  train_acc, train_loss, test_acc, test_loss = populate_metrics_last_E(
    DS, layer, search_param, scales, [run], TRUNC_field=None)
  trunc_train_acc, trunc_train_loss, trunc_test_acc, trunc_test_loss = populate_metrics_last_E(
    DS, layer, search_param, scales, [run], TRUNC_field=FIELD_short)


  alpha = np.zeros((len(scales), 2))
  for scale in scales:
    model_name = f"SETOL/{DS}/{layer}/{search_param}_{2**scale}"
    E = last_epoch(run, model_name)
    details = Trainer.load_details(run, model_name).query(f"epoch == {E}")
    alpha[scale,:] = details.loc[:, "alpha"]
    
  X = [2**s for s in scales]
  plot_one = lambda ax, Y, label: ax.plot(X, Y, '-', label = r"$\Delta$" + label)

  # Error = 1 - accuracy
  plot_one(axes[0], train_acc - trunc_train_acc, f"train error")
  plot_one(axes[0], test_acc  - trunc_test_acc,  f"test error")
  
  plot_one(axes[1], trunc_train_loss - train_loss, f"train loss")
  plot_one(axes[1], trunc_test_loss  - test_loss,  f"test loss")

  if layer in ("all", "FC1", "FC1_WHITENED"): axes[2].plot(X, alpha[:,0], label=r"FC1 $\alpha$", color="red")
  if layer in ("all", "FC2", "FC2_WHITENED"): axes[2].plot(X, alpha[:,1], label=r"FC2 $\alpha$", color="green")
  

  common_title = f"{search_param} search trained layer(s): {layer} at final epoch"
  axes[0].set(xlabel = f"{search_param} factor", ylabel = r"$\Delta$ error", title=f"{common_title}\ntruncated error - train error")
  axes[1].set(xlabel = f"{search_param} factor", ylabel = r"$\Delta$ loss",  title=f"{common_title}\ntruncated loss - train loss")
  axes[2].set(xlabel = f"{search_param} factor", ylabel = r"$\alpha$",  title=f"{common_title}\n" + r"$\alpha$ by layer")

  axes[2].set(ylim=(1.4, 2.3))
  
  for ax in axes: ax.legend()

  if save_dir is not None:
    if not isinstance(save_dir, Path): save_dir = Path(save_dir)
    if not save_dir.exists(): save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"mlp3_trunc_error_by_{search_param}_run_{run}_{layer}_{FIELD_short}.png", bbox_inches='tight')



def plot_truncated_errors_by_metric(DS, layer, search_param, scales, runs,
    layers,
    WW_metric,      # WW_metric to plot
    XMIN=True,      # Whether to use xmin, or detX as the tail demarcation
    save_dir = None,
  ):
  FIELD = 'xmin' if XMIN else "detX_val_unrescaled"
  FIELD_short = ['detX', 'xmin'][XMIN]

  L = len(layers)
  fig, axes = plt.subplots(ncols = L, nrows = 1, figsize=(6*L + 1*(L-1), 4))
  set_styles()
  if L == 1: axes = [axes]

  train_acc, train_loss, test_acc, test_loss = populate_metrics_last_E(
    DS, layer, search_param, scales, runs, TRUNC_field=None, FLAT=True)
  trunc_train_acc, trunc_train_loss, trunc_test_acc, trunc_test_loss = populate_metrics_last_E(
    DS, layer, search_param, scales, runs, TRUNC_field=FIELD_short, FLAT=True)

  WW_data = populate_WW_metric_last_epoch(DS, layer, search_param, scales, runs, WW_metric).reshape((-1, 2))

  WW_data = WW_data
  for l, ax in zip(layers, axes):
    common_title = f"{search_param} search trained layer(s): {layer} at last epoch"
    layer_name = ["FC1", "FC2"][l]

    ax.plot(WW_data[:, l], train_acc - trunc_train_acc, '+', label="truncated train error - train error")
    ax.plot(WW_data[:, l],  test_acc -  trunc_test_acc, '+', label="truncated test error - test error")
    ax.plot(WW_data[:, l], train_acc -        test_acc, '+', label="train error - test error")

    ax.set(xlabel = WW_metric, ylabel=r"$\Delta$ error", title=f"{common_title}\ntrained layer {layer_name}")
    ax.legend()
    ax.axhline(0, color="gray", zorder=-1)

  if save_dir is not None:
    if not isinstance(save_dir, Path): save_dir = Path(save_dir)
    if not save_dir.exists(): save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"mlp3_trunc_error_by_{search_param}_{WW_metric}_{layer}_{FIELD_short}.png", bbox_inches='tight')