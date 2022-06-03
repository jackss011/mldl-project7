import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import os
from datetime import datetime

from datasets import RODDataset
from utils import get_epochs_in_model_folder, select_device
import networks as n
from train import HP, setup_hp_arguments


BATCH_SIZE = 64
LOADER_WORKERS = 8


def eval(models_folder, every=1, results_file=None, desc="", hp=None):
  device = select_device()

  # ======= DATA ========
  ds_eval = RODDataset("data", train=False, image_size=224)
  dl_eval = DataLoader(ds_eval, batch_size=BATCH_SIZE, pin_memory=True, num_workers=LOADER_WORKERS)


  # ======= MODELS ========
  model_rgb = n.FeatureExtractor().to(device)
  model_d = n.FeatureExtractor().to(device)

  def combine_modes(rgb, d) -> torch.Tensor:
    return torch.cat((model_rgb(rgb), model_d(d)), dim=1)

  model_task = n.RecognitionClassifier(512*2, 51).to(device)

  model_list = [model_rgb, model_d, model_task]


  # ======= LOADING ========
  def load_models(path):
    l = torch.load(path)
    model_rgb.load_state_dict(l['model_rgb'])
    model_d.load_state_dict(l['model_d'])
    model_task.load_state_dict(l['model_task'])


  # ======= EVAL LOOP ========
  def eval_epoch(n):
    for m in model_list:
      m.eval()
    
    correct = 0
    total = 0

    for rgb, d, gt in tqdm(dl_eval, desc=f"Epoch {n}"):
      rgb, d, gt = rgb.to(device), d.to(device), gt.to(device)

      f = combine_modes(rgb, d)
      pred = F.softmax(model_task(f), dim=1)

      correct += (pred.argmax(dim=1) == gt).sum().item()
      total += gt.size(0)
    
    accuracy = correct/total

    return accuracy, correct, total


  # ===== START EVALUATION =======
  print(f"\n====> EVALUATING: {hp}")
  print("\n-- Snapshots folder:", models_folder)

  epochs = get_epochs_in_model_folder(models_folder)
  selected_epochs = list(reversed(epochs))[slice(0, len(epochs), every)]

  print(f"\n++ Evaluating every {every} epochs:", selected_epochs, '\n')

  for e in selected_epochs:
    # load model
    model_path = os.path.join(models_folder, f"model_{e}.tar")
    load_models(model_path)

    accuracy, correct, total = eval_epoch(e)

    print(f"RESUTLS: {accuracy*100:.1f}% ({correct}/{total})\n")

    if results_file:
      with open(results_file, 'a') as f:
        now = datetime.now()
        date = now.strftime('%b%d_%H-%M-%S')
        f.write(f"{date}, {desc}, {e}, {accuracy:.4f}, {correct}, {total}\n")




# ++++ START ++++
if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  setup_hp_arguments(parser)
  args = parser.parse_args()

  hp = HP()
  hp.load_from_args(args)

  hp_folder = hp.to_filename()
  models_folder = os.path.join("snapshots", hp_folder)

  os.makedirs("results", exist_ok=True)
  eval(models_folder, hp=hp, results_file="results/log.csv", desc=hp_folder)