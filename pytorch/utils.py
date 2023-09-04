import numpy as np
from trainer import Trainer

def last_epoch(run, model_name):
  train_acc, _, _, _, _, _ = Trainer.load_metrics(run, model_name)
  if train_acc is None: return -1
  if np.sum(train_acc) == 0: return 0
  if np.min(train_acc[1:]) > 0: return len(train_acc) -1
  return np.argmin(train_acc[1:] > 0)


def metric_error_bars(DS, OPT, layer, scales, runs, search_param="BS"):
  all_metrics = []
  for scale in scales:
    model_name = f"SETOL/{DS}/{OPT}/{layer}/{search_param}_{2**scale}"
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

def average_DFS(model_name, runs, WW_metrics):
  import pandas as pd
  df_concat = pd.concat([
    Trainer.load_details(run, last_epoch(run, model_name), model_name).loc[:, WW_metrics]
    for run in runs
  ])
  means = df_concat.groupby(df_concat.index).mean()
  stdevs = df_concat.groupby(df_concat.index).std()

  return means, stdevs

def DF_error_bars(DS, OPT, layer, scales, runs, WW_metrics, search_param="BS"):
  mean_DFs, stdev_DFs = zip(*[
    average_DFS(f"SETOL/{DS}/{OPT}/{layer}/{search_param}_{2**scale}", runs, WW_metrics)
    for scale in scales
  ])

  return mean_DFs, stdev_DFs



def aggregate_DFs(DS, OPT, runs, run_name, layers = None):
  if layers is None: layers = [
    "FC1", "FC2", "all", "FC1_WHITENED", "FC2_WHITENED"
  ]

  DF = None
  for layer in layers:
    for scale in range(6):
      model_name = f"SETOL/{DS}/{OPT}/{layer}/BS_{2**scale}"
  
      for run in runs:
        train_acc, train_loss, test_acc, test_loss = Trainer.load_metrics(run, model_name)
        if train_acc is None: continue
        for epoch in range(1, last_epoch(run, model_name)+1):
          details = Trainer.load_details(run, epoch, model_name)
          details['trained_layer'] = layer
          details['batch_size'] = 2**scale
          details['epoch'] = epoch
  
          details['train_acc']  = train_acc[epoch]
          details['train_loss'] = train_loss[epoch]
          details['test_acc']   = test_acc[epoch]
          details['test_loss']  = test_loss[epoch]
  
          if DF is None: DF = details
          else:          DF = DF.append(details)

  return DF
