import numpy as np
import pandas as pd

from trainer import Trainer

def last_epoch(run, model_name):
  train_acc, _, _, _, _, _ = Trainer.load_metrics(run, model_name)
  if train_acc is None: return -1
  if np.sum(train_acc) == 0: return 0
  if np.min(train_acc[1:]) > 0: return len(train_acc) -1
  return np.argmin(train_acc[1:] > 0)


def metric_error_bars(DS, layer, scales, runs, search_param="BS"):
  all_metrics = []
  for scale in scales:
    model_name = f"SETOL/{DS}/{layer}/{search_param}_{2**scale}"
    metrics = [
      [ m[last_epoch(run, model_name)] for m in Trainer.load_metrics(run, model_name)]
      for run in runs
    ]

    # Transpose, take mean
    metrics = [ (np.mean(metric), np.std(metric)) for metric in zip(*metrics) ]

    # Transpose again
    all_metrics.append(tuple(zip(*metrics)))

  means, stdevs = tuple(zip(*all_metrics))
  return means, stdevs


def average_DFS_last_epoch(model_name, runs, WW_metrics):
  df_concat = pd.concat([
    Trainer.load_details(run, model_name).query(f"epoch == {last_epoch(run, model_name)}").loc[:, WW_metrics]
    for run in runs
  ])
  means = df_concat.groupby(df_concat.index).mean()
  stdevs = df_concat.groupby(df_concat.index).std()

  return means, stdevs

def DF_error_bars(DS, layer, scales, runs, WW_metrics, search_param="BS"):
  mean_DFs, stdev_DFs = zip(*[
    average_DFS_last_epoch(f"SETOL/{DS}/{layer}/{search_param}_{2**scale}", runs, WW_metrics)
    for scale in scales
  ])

  return mean_DFs, stdev_DFs


def aggregate_DFs(DS, search_param, layer, scale, runs):
  model_name = f"SETOL/{DS}/{layer}/{search_param}_{2**scale}"

  for run in runs:
    details_path = Trainer.details_path(run, model_name)
    if details_path.exists():
      print(f"found {details_path}. skipping")
      continue

    train_acc, train_loss, val_acc, val_loss, test_acc, test_loss = Trainer.load_metrics(run, model_name, True)
    if train_acc is None:
      print(f"No metrics found for {model_name} {run}")
      return

    DF = None
    E = last_epoch(run, model_name)
    for epoch in range(1, E+1):
      details = pd.read_pickle(Trainer.save_dir(run, epoch, model_name) / "WW_details")
      details['epoch'] = epoch
      details['run_number'] = run
      details['model_name'] = model_name

      details['train_acc']  = train_acc[epoch]
      details['train_loss'] = train_loss[epoch]
      details['val_acc']    = val_acc[epoch]
      details['val_loss']   = val_loss[epoch]
      details['test_acc']   = test_acc[epoch]
      details['test_loss']  = test_loss[epoch]

      if DF is None: DF = details
      else:          DF = DF.append(details)

    DF.to_pickle(details_path)


def populate_metrics_all_epochs(DS, trained_layer, search_param, scale, runs, TRUNC_field=None):
  model_name = f"SETOL/{DS}/{trained_layer}/{search_param}_{2**scale}"

  Emin = min(last_epoch(run, f"SETOL/{DS}/{trained_layer}/{search_param}_{2**scale}") for run in runs)
  train_acc        = np.zeros((len(runs), Emin+1))
  train_loss       = np.zeros((len(runs), Emin+1))
  test_acc         = np.zeros((len(runs), Emin+1))
  test_loss        = np.zeros((len(runs), Emin+1))


  save_file=None
  for run in runs:
    if TRUNC_field is not None:
      save_file = f"{TRUNC_field}_truncated_accuracy_run_{run}.npy"

    metrics = Trainer.load_metrics(run, model_name, save_file=save_file)
    if metrics[0] is None:
      print(f"metrics for {model_name} {save_file} not found")
      continue

    train_acc[run,:Emin+1], train_loss[run,:Emin+1], _, _, test_acc[run,:Emin+1], test_loss[run,:Emin+1] = [
      m[:Emin+1]
      for m in metrics
    ]

  return train_acc, train_loss, test_acc, test_loss


def populate_metrics_last_epoch(DS, trained_layer, search_param, scales, runs, TRUNC_field=None, FLAT=False):
  train_acc   = np.zeros((len(scales), len(runs)))
  train_loss  = np.zeros((len(scales), len(runs)))
  test_acc    = np.zeros((len(scales), len(runs)))
  test_loss   = np.zeros((len(scales), len(runs)))

  save_file = None
  for scale_i, scale in enumerate(scales):
    model_name = f"SETOL/{DS}/{trained_layer}/{search_param}_{2**scale}"
    for run_i, run in enumerate(runs):
      if TRUNC_field is not None:
        save_file = f"{TRUNC_field}_truncated_accuracy_run_{run}.npy"

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



def populate_WW_metric_all_epochs(DS, trained_layer, search_param, scale, runs, Emin, WW_metric):
  model_name = f"SETOL/{DS}/{trained_layer}/{search_param}_{2**scale}"

  WW_data = np.zeros((len(runs), Emin+1, 2))

  for run_i, run in enumerate(runs):
    details = Trainer.load_details(run, model_name)
    for epoch in range(1, Emin+1):
      WW_data[run_i, epoch, :] = details.query(f"epoch == {epoch}").loc[:, WW_metric]
  
  return WW_data



def populate_WW_metric_last_epoch(DS, trained_layer, search_param, scales, runs, WW_metric):
  WW_data = np.zeros((len(scales), len(runs), 2))

  for scale_i, scale in enumerate(scales):
    model_name = f"SETOL/{DS}/{trained_layer}/{search_param}_{2**scale}"
    for run_i, run in enumerate(runs):
      E = last_epoch(run, model_name)

      WW_data[scale_i, run_i, :] = Trainer.load_details(
        run, model_name).query(f"epoch == {E}").loc[:, WW_metric]
  
  return WW_data


