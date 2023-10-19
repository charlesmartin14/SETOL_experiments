from pathlib import Path

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn import linear_model
from weightwatcher import WeightWatcher

from trainer import Trainer
from utils import last_epoch

from utils import metric_error_bars, DF_error_bars
from utils import populate_metrics_all_epochs, populate_metrics_last_epoch
from utils import populate_WW_metric_all_epochs, populate_WW_metric_last_epoch


# Set standard constants
SMALL_SIZE = 12
MEDIUM_SIZE = 16
LARGE_SIZE = 30

plt.rc('font', size=LARGE_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('legend', title_fontsize=MEDIUM_SIZE) # legend title fontsize
plt.rc('figure', titlesize=LARGE_SIZE)  # fontsize of the figure title


def save_fig(save_dir, save_file, fig):
  if save_dir is None: return
  if not isinstance(save_dir, Path): save_dir = Path(save_dir)
  if not save_dir.exists(): save_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(save_dir / save_file, bbox_inches='tight')

def make_colors(search_param, scales):
  blue_colors = plt.cm.Blues(np.linspace(0.5, 1, len(scales)))
  green_colors = plt.cm.Greens(np.linspace(0.5, 1, len(scales)))
  red_colors = plt.cm.Reds(np.linspace(0.5, 1, len(scales)))
  if search_param == "BS": blue_colors[ 0] = green_colors[ 0] = red_colors[0]
  if search_param == "LR": blue_colors[-1] = green_colors[-1] = red_colors[0]
  return red_colors, green_colors, blue_colors


def plot_loss(DS, trained_layer, search_param, scale, runs, plot_layer, WW_metric,
  LOSS = True,
  xlim=None,
  ylim=None,
  save_dir=None,
  E0=4
):
  model_name = f"SETOL/{DS}/{trained_layer}/{search_param}_{2**scale}"

  runs = [
    run for run in runs
    if Trainer.save_dir(run, 0, model_name).exists()
  ]
  if not runs: return

  if ylim is None: ylim = (0, None)
  if ylim[0] is None: ylim = (0, ylim[1])

  if xlim is None: xlim = (2, None)
  if xlim[0] is None: xlim = (2, xlim[1])
  
  layer_name = ["FC1", "FC2"][plot_layer]
  trained_layers = {"all": "all layers trained", "FC1": "only FC1 trained", "FC2": "only FC2 trained"}

  fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(14, 3))


  common_title = f"MLP3: {search_param}={2**scale}; {trained_layers[trained_layer]}"

  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,
                    wspace=0.4,
                    hspace=0.5)
  for ax, TRAIN in zip(axes, [True, False]):
    y_ax_name = f"{'TRAIN' if TRAIN else 'TEST'} {'loss' if LOSS else 'error'}"
    for run in runs:
      train_acc, train_loss, _, _, test_acc, test_loss = Trainer.load_metrics(run, model_name)
      X = np.arange(E0, last_epoch(run, model_name))

      metric_vals = np.zeros(X.shape)
      details = Trainer.load_details(run, model_name)
      for e in X:
        metric_vals[e-E0] = details.query(f"epoch == {e}").loc[plot_layer, WW_metric]
      if LOSS:
        if TRAIN: ax.plot(metric_vals, train_loss[X], '-', label=f"seed={run+1}", alpha=0.5)
        else:     ax.plot(metric_vals, test_loss [X], '-', label=f"seed={run+1}", alpha=0.5)
      else:
        if TRAIN: ax.plot(metric_vals, 1-train_acc[X], '-', label=f"seed={run+1}", alpha=0.5)
        else:     ax.plot(metric_vals, 1-test_acc [X], '-', label=f"seed={run+1}", alpha=0.5)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    ax.legend() #bbox_to_anchor=(1.45, 0.75))

    ax.set(title=f"{common_title}\n{y_ax_name} vs. {WW_metric} for layer {layer_name}",
          ylabel=y_ax_name, xlabel=WW_metric, xlim=xlim, ylim=ylim)
    
  LOSS = 'loss' if LOSS else 'error'
  save_fig(save_dir, f"mlp3_{LOSS}_by_{search_param}={2**scale}_{trained_layer}_{layer_name}.png", fig)


def plot_loss_binned(DS, trained_layer, search_param, scales, runs, plot_layer, WW_metric,
  LOSS = True,
  precision = 0.05,
  xlim=None,
  ylim=None,
  save_dir=None,
  E0=4
):
  model_name = f"SETOL/{DS}/{trained_layer}/{search_param}_{2**scales[0]}"

  runs = [
    run for run in runs
    if Trainer.save_dir(run, 0, model_name).exists()
  ]
  if not runs: return
  
  if ylim is None: ylim = (0, None)
  if xlim is None: xlim = (None, None)
  
  layer_name = ["FC1", "FC2"][plot_layer]
  trained_layers = {"all": "all layers trained", "FC1": "only FC1 trained", "FC2": "only FC2 trained"}

  fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(14, 3))
  red_colors, green_colors, blue_colors = make_colors(search_param, scales)

  common_title = f"MLP3: various {search_param}s; {trained_layers[trained_layer]}"
  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,
                    wspace=0.4,
                    hspace=0.5)

  ERR_TYPE = 'loss' if LOSS else 'error'
  WW_metrics = ["train_acc", "train_loss", "test_acc", "test_loss", WW_metric, "epoch", "run_number"]

  # When plotting alpha, also plot a dashed linear fit for points above alpha=2
  def plot_lm_fit(df, y_col_name, scale):
    lm = linear_model.LinearRegression()
    X = np.sort(pd.unique(df.alpha)).reshape((-1,1))
    df = df.query(f"alpha >= 2").loc[:, ("alpha", y_col_name)].values
    lm.fit(df[:, 0].reshape((-1,1)), df[:, 1].reshape((-1,1)))
    label = r"linear fit $\alpha>2$" if scale == scales[0] else None
    ax.plot(X, lm.predict(X), '-.', color="black", zorder=10, label=label)


  for ax, TRAIN in zip(axes, [True, False]):
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

    MODE = 'TRAIN' if TRAIN else 'TEST'
    y_ax_name = f"{MODE} {ERR_TYPE}"
    y_col_name = f"{MODE.lower()}_{'loss' if LOSS else 'acc'}"

    colors = blue_colors if TRAIN else green_colors
    all_df = None

    for scale, color in zip(scales, colors):
      model_name = f"SETOL/{DS}/{trained_layer}/{search_param}_{2**scale}"
      df = pd.concat([
        Trainer.load_details(run, model_name).loc[plot_layer, WW_metrics]
        for run in runs
      ])
      df = df.query(f"{E0} < epoch")
      if xlim[0] is not None: df = df.query(f"{xlim[0]} <= {WW_metric}")
      if xlim[1] is not None: df = df.query(f"{xlim[1]} >= {WW_metric}")

      if not LOSS: df.loc[:, y_col_name] = 1 - df.loc[:, y_col_name]

      if all_df is None:  all_df = df.copy()
      else:               all_df = pd.concat([all_df, df]) 

      # Round to specified precision for binning.
      df.loc[:, WW_metric] = df.loc[:, WW_metric] / precision
      df = df.round(pd.Series([0], index=[WW_metric]))
      df.loc[:, WW_metric] = df.loc[:, WW_metric] * precision

      # Plot error bars
      X = np.sort(pd.unique(df.loc[:, WW_metric]))
      Y    = df.groupby(WW_metric).mean().loc[:,y_col_name]
      yerr = df.groupby(WW_metric).std ().loc[:,y_col_name]
      ax.errorbar(X, Y, xerr = 0, yerr=yerr, label=f"{search_param}={2**scale}", color=color)

      if WW_metric == "alpha" and trained_layer == "all":
        plot_lm_fit(df, y_col_name, scale)

    if WW_metric == "alpha" and trained_layer != "all":
      plot_lm_fit(all_df, y_col_name, scales[0])

    title = f"{common_title}\n{MODE} {ERR_TYPE} vs. {WW_metric} for layer {layer_name}"
    ax.set(ylabel=y_ax_name, xlabel=WW_metric, title=title) #, xlim=xlim, ylim=ylim)
    ax.legend(bbox_to_anchor=(1.6, 0.75))
    if WW_metric == "alpha": ax.axvline(2, linewidth=0.5, zorder=-1, color="red")

  save_fig(save_dir, f"mlp3_{ERR_TYPE}_by_{search_param}_{trained_layer}_{layer_name}_binned.png", fig)


def plot_by_scales(DS, trained_layer, scales, runs, WW_metrics,
  plot_layer = 0,
  search_param="BS",
  save_dir = None,
):
  red_colors, green_colors, blue_colors = make_colors(search_param, scales)

  fig, axes = plt.subplots(nrows=1, ncols=len(WW_metrics)+2, figsize=(8*(2+len(WW_metrics)) + 3, 4))
  axes[-1].axis('off')

  means, stdevs = metric_error_bars(DS, trained_layer, scales, runs, search_param=search_param)
  train_acc, train_loss, _, _, test_acc, test_loss = tuple(zip(*means))
  train_acc_SD, train_loss_SD, _, _, test_acc_SD, test_loss_SD = tuple(zip(*stdevs))

  mean_DFs, stdev_DFs = DF_error_bars(DS, trained_layer, scales, runs, WW_metrics, search_param=search_param)

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

    ax.set(title=f"MLP3: {WW_metric} for {layer_name} vs. train/test error\nVarious {search_param_long}s considered",
      xlabel= WW_metric, ylabel="error")

    plot_legends(ax, tr_labels, te_labels)

  plt.tight_layout(rect=[0, 0, 0.70, 1.4]) 
  plt.subplots_adjust(wspace=0.95, hspace=0.95)
  
  save_fig(save_dir, f"mlp3_quality_by_{search_param}_{trained_layer}_{layer_name}.png", fig)



def plot_over_epochs(DS, trained_layer, search_param, scale, runs, WW_metric, plot_layers):
  """ Plots a particular WW metric, such as "alpha", over the epochs of a series of training runs. Each trained layer is shown in a separate column
      DS, layer, search_param, scale: Fields that identify which experiment was done.
      WW_metric: A WeightWatcher metric to be plotted. 
      layers: A list of indices of layers for which WW_metric should be plotted. Valid indices are {0, 1}
  """
  model_name = f"SETOL/{DS}/{trained_layer}/{search_param}_{2**scale}"

  fig, axes = plt.subplots(nrows=1, ncols=len(plot_layers), figsize = (6*len(plot_layers), 4))

  if len(plot_layers) == 1: axes = [axes]

  for run in runs:
    details = Trainer.load_details(run, model_name)
    E = last_epoch(run, model_name)
    for l, ax in zip(plot_layers, axes):
      metric_data = np.zeros((E,))
      for e in range(E):
        metric_data[e] = details.query(f'epoch == {e+1}').loc[l, WW_metric]
      ax.plot(metric_data, '+', label = f"seed = {run+1}")

  for l, ax in zip(plot_layers, axes):
    ax.set(title=f"{model_name}\nlayer FC{l+1}", xlabel="epoch", ylabel=WW_metric)
    ax.legend()



def plot_truncated_accuracy_over_epochs(DS, trained_layer, search_param, scale, runs,
    XMIN=True,      # Whether to use xmin, or detX as the tail demarcation
    ylims=None,     # Maximum values to set for each plot so they can be compared
    E0=4, # Skip the first few epochs so that the variance is contained.
    save_dir=None,  # place to save the images
  ):
  fig, axes = plt.subplots(ncols = 3, nrows = 1, figsize=(20, 4))

  if ylims is None: ylims = (None, None)

  FIELD = 'xmin' if XMIN else "detX_val_unrescaled"
  FIELD_short = ['detX', 'xmin'][XMIN]

  train_acc, train_loss, test_acc, test_loss = populate_metrics_all_epochs(
    DS, trained_layer, search_param, scale, runs, TRUNC_field=None)

  trunc_train_acc, trunc_train_loss, trunc_test_acc, trunc_test_loss = populate_metrics_all_epochs(
    DS, trained_layer, search_param, scale, runs, TRUNC_field=FIELD_short)
  
  Emin = train_acc.shape[1]-1

  alpha = populate_WW_metric_all_epochs(DS, trained_layer, search_param, scale, runs, Emin, "alpha")

  X = np.arange(E0, Emin+1)
  plot_one = lambda ax, Y, label, color=None: ax.errorbar(X, np.mean(Y[:,E0:], axis=0),
    yerr=np.std(Y[:,E0:], axis=0), fmt='-', label=r'$\Delta$ ' + label, color=color)

  # Error = 1 - accuracy
  plot_one(axes[0], train_acc        - trunc_train_acc, "train error")
  plot_one(axes[0], test_acc         - trunc_test_acc , "test error")
  test_err = 1 - test_acc[:, E0:Emin+1]
  axes[0].errorbar(X, np.mean(test_err, axis=0), yerr=np.std(test_err, axis=0),
    fmt='-', label="plain test error")

  plot_one(axes[1], trunc_train_loss - train_loss     , "train loss")
  plot_one(axes[1], trunc_test_loss  - test_loss      , "test loss")

  if trained_layer in ("all", "FC1", "FC1_WHITENED"): plot_one(axes[2], alpha[:,:,0], label=r"FC1 $\alpha$", color="red")
  if trained_layer in ("all", "FC2", "FC2_WHITENED"): plot_one(axes[2], alpha[:,:,1], label=r"FC2 $\alpha$", color="green")


  common_title = f"{search_param} = {2**scale} trained layer(s): {trained_layer}"
  axes[0].set(xlabel="epochs", ylabel=r"$\Delta$ error", title=f"{common_title}\ntruncated error difference")
  axes[1].set(xlabel="epochs", ylabel=r"$\Delta$ loss",  title=f"{common_title}\ntruncated train loss - train loss")
  axes[2].set(xlabel="epochs", ylabel=r"$\alpha$",       title=f"{common_title}\n" + r"$\alpha$ for trained layers")

  axes[2].axhline(2, color="gray", zorder=-1)

  axes[0].set(ylim=(-0.0075, ylims[0]))
  axes[1].set(ylim=(-0.02, ylims[1]))
  axes[2].set(ylim=(1, None))
  for ax in axes:
    ax.legend()
    ax.axhline(0, color="gray", zorder=-1)

  save_fig(save_dir, f"mlp3_trunc_error_by_epochs_{search_param}_{scale}_{trained_layer}_{FIELD_short}.png", fig)


def plot_truncated_errors_by_scales(DS, trained_layer, search_param, scales, run,
    XMIN=True,      # Whether to use xmin, or detX as the tail demarcation
    save_dir = None,
  ):
  FIELD = 'xmin' if XMIN else "detX_val_unrescaled"
  FIELD_short = ['detX', 'xmin'][XMIN]

  fig, axes = plt.subplots(ncols = 3, nrows = 1, figsize=(20, 2))

  train_acc, train_loss, test_acc, test_loss = populate_metrics_last_epoch(
    DS, trained_layer, search_param, scales, [run], TRUNC_field=None)
  trunc_train_acc, trunc_train_loss, trunc_test_acc, trunc_test_loss = populate_metrics_last_epoch(
    DS, trained_layer, search_param, scales, [run], TRUNC_field=FIELD_short)


  alpha = np.zeros((len(scales), 2))
  for scale in scales:
    model_name = f"SETOL/{DS}/{trained_layer}/{search_param}_{2**scale}"
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

  if trained_layer in ("all", "FC1", "FC1_WHITENED"): axes[2].plot(X, alpha[:,0], label=r"FC1 $\alpha$", color="red")
  if trained_layer in ("all", "FC2", "FC2_WHITENED"): axes[2].plot(X, alpha[:,1], label=r"FC2 $\alpha$", color="green")
  

  common_title = f"{search_param} search trained layer(s): {trained_layer} at final epoch"
  axes[0].set(xlabel = f"{search_param} factor", ylabel = r"$\Delta$ error", title=f"{common_title}\ntruncated error - train error")
  axes[1].set(xlabel = f"{search_param} factor", ylabel = r"$\Delta$ loss",  title=f"{common_title}\ntruncated loss - train loss")
  axes[2].set(xlabel = f"{search_param} factor", ylabel = r"$\alpha$",  title=f"{common_title}\n" + r"$\alpha$ by layer")

  axes[2].set(ylim=(1.4, 2.3))
  
  for ax in axes: ax.legend()

  save_fig(save_dir, f"mlp3_trunc_error_by_{search_param}_run_{run}_{trained_layer}_{FIELD_short}.png", fig)



def plot_truncated_errors_by_metric(DS, trained_layer, search_param, scales, runs,
    plot_layers,
    WW_metric,      # WW_metric to plot
    XMIN=True,      # Whether to use xmin, or detX as the tail demarcation
    save_dir = None,
  ):
  FIELD = 'xmin' if XMIN else "detX_val_unrescaled"
  FIELD_short = ['detX', 'xmin'][XMIN]

  L = len(plot_layers)
  fig, axes = plt.subplots(ncols = L, nrows = 1, figsize=(6*L + 1*(L-1), 4))
  if L == 1: axes = [axes]

  train_acc, train_loss, test_acc, test_loss = populate_metrics_last_epoch(
    DS, trained_layer, search_param, scales, runs, TRUNC_field=None, FLAT=True)
  trunc_train_acc, trunc_train_loss, trunc_test_acc, trunc_test_loss = populate_metrics_last_epoch(
    DS, trained_layer, search_param, scales, runs, TRUNC_field=FIELD_short, FLAT=True)

  WW_data = populate_WW_metric_last_epoch(DS, trained_layer, search_param, scales, runs, WW_metric).reshape((-1, 2))

  WW_data = WW_data
  for l, ax in zip(plot_layers, axes):
    common_title = f"{search_param} search trained layer(s): {trained_layer} at last epoch"
    layer_name = ["FC1", "FC2"][l]

    ax.plot(WW_data[:, l], train_acc - trunc_train_acc, '+', label="truncated train error - train error")
    ax.plot(WW_data[:, l],  test_acc -  trunc_test_acc, '+', label="truncated test error - test error")
    ax.plot(WW_data[:, l], train_acc -        test_acc, '+', label="train error - test error")

    ax.set(xlabel = WW_metric, ylabel=r"$\Delta$ error", title=f"{common_title}\ntrained layer {layer_name}")
    ax.legend()
    ax.axhline(0, color="gray", zorder=-1)

  save_fig(save_dir, f"mlp3_trunc_error_by_{search_param}_{WW_metric}_{trained_layer}_{FIELD_short}.png", fig)


def plot_ww_metrics_by_scales(DS, trained_layer, search_param, scales, runs,
    plot_layers,
    WW_metrics,     # WW_metrics to plot
    save_dir = None,
  ):
  assert len(WW_metrics) == 2, len(WW_metrics)
  assert len(plot_layers) <= 2, len(plot_layers)

  metric_data1 = populate_WW_metric_last_epoch(DS, trained_layer, search_param, scales, runs, WW_metrics[0]) #.reshape((-1, 2))
  metric_data2 = populate_WW_metric_last_epoch(DS, trained_layer, search_param, scales, runs, WW_metrics[1]) #.reshape((-1, 2))

  L = len(plot_layers)
  fig, axes = plt.subplots(ncols = L, nrows = 1, figsize=(7*L-1, 4))
  if L == 1: axes = [axes]

  layer_names = ["FC1", "FC2"]
  search_param_long = { "BS": "batch size", "LR": "learning rate"}[search_param]
  common_title = f"MLP3: {WW_metrics[0]} vs {WW_metrics[1]}\nVarious {search_param_long}s considered"
  for l, ax in zip(plot_layers, axes):
    for scale_i, scale in enumerate(scales):
      model_name = f"SETOL/{DS}/{trained_layer}/{search_param}_{2**scale}"
      X = metric_data1[scale_i, :, l]
      Y = metric_data2[scale_i, :, l]
      ax.errorbar(np.mean(X), np.mean(Y), xerr=np.std(X, axis=0), yerr=np.std(Y, axis=0), fmt='-', label=f"{search_param}={2**scale}")
    ax.set(xlabel=WW_metrics[0], ylabel=WW_metrics[1], title=f"{common_title}\nLayer {layer_names[l]}")
    ax.legend()

  save_fig(save_dir, f"mlp3_{WW_metrics[0]}_by_{WW_metrics[1]}_{search_param}_{trained_layer}.png", fig)


def plot_detX(DS, trained_layer, search_param, scales, runs, plot_layers, save_dir = None,):
  red_colors, green_colors, blue_colors = make_colors(search_param, scales)

  alpha = populate_WW_metric_last_epoch(DS, trained_layer, search_param, scales, runs, "alpha")
  # xmin = populate_WW_metric_last_epoch(DS, layer, search_param, scales, runs, "xmin")
  # detX = populate_WW_metric_last_epoch(DS, layer, search_param, scales, runs, "detX_val")

  Dl = populate_WW_metric_last_epoch(DS, trained_layer, search_param, scales, runs, "detX_delta")

  L = len(plot_layers)
  fig, axes = plt.subplots(ncols=L, nrows=1, figsize=(9*L - 2, 4))

  layer_names = ["FC1", "FC2"]
  search_param_long = { "BS": "batch size", "LR": "learning rate"}[search_param]

  deltaL = r"$\Delta \lambda_{min}$"
  common_title = f"MLP3: {deltaL} (PL fit) - (|detX|=1) \nVarious {search_param_long}s considered"

  if L == 1: axes = [axes]


  for l, ax in zip(plot_layers, axes):
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

    for scale in scales:
      ax.plot(alpha[scale, :, l], Dl[scale, :, l], '.', color=blue_colors[scale], label=f"{search_param}={2**scale}")

    ax.axvline(2, color="red", linewidth=0.5, zorder=-1)
    ax.axhline(0, color="red", linewidth=0.5, zorder=-1)

    ax.set(xlabel = r"PL $\alpha$", ylabel=r"$\Delta \lambda_{min}$",
      xlim=(1.85, None), ylim=(min(-0.1, np.min(Dl[:,l] * 1.2)), None),
      title=f"{common_title}\nlayer {layer_names[l]}")
    ax.legend(bbox_to_anchor=(1.35, 0.75))

  save_fig(save_dir, f"mlp3_detX_delta_{search_param}_{trained_layer}.png", fig)



