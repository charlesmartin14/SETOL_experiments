import numpy as np
from trainer import Trainer

def last_epoch(run, model_name):
    train_acc, train_loss, test_acc, test_loss = Trainer.load_metrics(run, model_name)
    if train_acc is None: return -1
    if np.sum(train_acc) == 0: return 0
    if np.min(train_acc[1:]) > 0: return len(train_acc) -1
    return np.argmin(train_acc[1:] > 0)


def aggregate_DFs(DS, opt, runs, run_name, layers = None):
  if layers is None: layers = [
    "layer0", "layer1", "all", "layer0_WHITENED", "layer1_WHITENED"
  ]

  DF = None
  for layer in layers:
    model_name = f"SETOL/{DS}/{opt}/{layer}"

    for run in runs:
      train_acc, train_loss, test_acc, test_loss = Trainer.load_metrics(run, model_name)
      for epoch in range(1, last_epoch(run, model_name)+1):
        details = Trainer.load_details(run, epoch, model_name)
        details['trained_layer'] = layer
        details['batch_size'] = run_name(run)
        details['epoch'] = epoch

        details['train_acc']  = train_acc[epoch]
        details['train_loss'] = train_loss[epoch]
        details['test_acc']   = test_acc[epoch]
        details['test_loss']  = test_loss[epoch]

        if DF is None: DF = details
        else:          DF = DF.append(details)

  return DF
