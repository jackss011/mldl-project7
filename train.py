import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from tqdm import tqdm
import os
import math
from datetime import datetime
import dataclasses
from dataclasses import dataclass

from datasets import PretextSynRODDataset, PretextRODDataset, RODDataset, SynRODDataset
from networks import FeatureExtractor, RecognitionClassifier, RotationClassifier, weight_init
from utils import LoaderIterator, ent_loss, get_epochs_in_model_folder, select_device, show_image

from torch.utils.tensorboard import SummaryWriter


TRAIN_WORKERS = 1   # workers used for loading training data (for each dataset)
EVAL_WORKERS = 4    # workers used for loading evaluation data (for each dataset)


# HYPERPARAMS
@dataclass
class HP:
  epochs: int = 10
  batch_size: int = 64
  lr: float = 1e-4
  momentum: float = 0.9
  weight_decay: float = 0.05
  pretext_weight: float = 1
  ent_weight: float = 0.1

  def to_filename(self):
    prs = [ f"{k}_{v}" for k, v in dataclasses.asdict(self).items() ]
    return "__".join(prs)

  def load_from_args(self, args):
    for f in dataclasses.fields(HP):
      hp_arg = vars(args)[f.name]
      if hp_arg is not None:
        setattr(self, f.name, hp_arg)


def setup_hp_arguments(parser):
  """
    Add hyperparameters to argparse so that they can be change via cmd.
    Example: `python train.py --lr 0.003` will override the default learning rate
  """
  parser.add_argument("--epochs", type=int, help="Number of epochs of training")
  parser.add_argument("--batch-size", type=int, help="Batch size")
  parser.add_argument("--lr", type=float, help="Learning rate for SGD")
  parser.add_argument("--momentum", type=float, help="Momentum for SGD")
  parser.add_argument("--weight-decay", type=float, help="Weight decay for SGD")
  parser.add_argument("-pw", "--pretext-weight", type=float, help="Pretext task loss weight")
  parser.add_argument("-ew", "--ent-weight", type=float, help="ENT loss weight")


# ==== RUN TRAINING ====
def run(hp: HP, resume=False, save_snapshots=True):
  """
    Performs a full training run with the specified hyperparameters.
    If resume is true tries to find the most recent snapshot with the same hps
  """
  device = select_device()

  print(f"\n+++ HPs:", hp)

  # ======= TENSORBOARD WRITER ========
  now = datetime.now()
  summary = SummaryWriter(log_dir=f"runs/{hp.to_filename()}/{now.strftime('%b%d_%H-%M-%S')}")
  # summary.add_hparams(dataclasses.asdict(hp), {})


  # ======= DATASETS ========
  ds_train_source = SynRODDataset("data", train=True, image_size=224)             # Source labelled
  ds_train_source_pt = PretextSynRODDataset("data", train=True, image_size=224)   # Source pretext
  ds_train_target_pt = PretextRODDataset("data", train=True, image_size=224)      # Target pretext
  ds_train_target = RODDataset("data", train=True, image_size=224)

  ds_eval_source = SynRODDataset("data", train=False, image_size=224)             # Test if can classify source
  ds_eval_source_pt = PretextSynRODDataset("data", train=False, image_size=224)   # Test if can predict rotation of source images
  ds_eval_target_pt = PretextRODDataset("data", train=False, image_size=224)       # Test if can predict rotation of target images
  ds_eval_target = RODDataset("data", train=False, image_size=224)


  # ======= DATALOADERS ========
  def dataloader_factory(ds, num_workers=TRAIN_WORKERS):
    return DataLoader(ds, batch_size=hp.batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)

  dl_train_source = dataloader_factory(ds_train_source)
  dl_train_source_pt = dataloader_factory(ds_train_source_pt)
  dl_train_target_pt = dataloader_factory(ds_train_target_pt)
  dl_train_target = dataloader_factory(ds_train_target)

  dl_eval_source = dataloader_factory(ds_eval_source, num_workers=EVAL_WORKERS)
  dl_eval_source_pt = dataloader_factory(ds_eval_source_pt, num_workers=EVAL_WORKERS)
  dl_eval_target_pt = dataloader_factory(ds_eval_target_pt, num_workers=EVAL_WORKERS)
  dl_eval_target = dataloader_factory(ds_eval_target, num_workers=EVAL_WORKERS)

  # ======= MODELS ========
  model_rgb = FeatureExtractor().to(device)
  model_d = FeatureExtractor().to(device)

  def combine_modes(rgb, d) -> torch.Tensor:
    return torch.cat((model_rgb(rgb), model_d(d)), dim=1)

  model_task = RecognitionClassifier(512*2, 47).to(device)
  model_pretext = RotationClassifier(512*2, 4).to(device)

  model_task.apply(weight_init)
  model_pretext.apply(weight_init)

  model_list = [model_rgb, model_d, model_task, model_pretext]

  criterion = torch.nn.CrossEntropyLoss().to(device)


  # ======= OPTIMIZERS ========
  def opt_factory(model):
    return SGD(model.parameters(), lr=hp.lr, momentum=hp.momentum, weight_decay=hp.weight_decay)

  opt_rgb = opt_factory(model_rgb)
  opt_d = opt_factory(model_d)
  opt_task = opt_factory(model_task)
  opt_pretext = opt_factory(model_pretext)

  opt_list = [opt_rgb, opt_d, opt_task, opt_pretext]


  # ======= SAVING ========
  def save_networks(path):
    torch.save({
      'model_rgb': model_rgb.state_dict(),
      'model_d': model_d.state_dict(),
      'model_task': model_task.state_dict(),
      'model_pretext': model_pretext.state_dict(),

      'opt_rgb': opt_rgb.state_dict(),
      'opt_d': opt_d.state_dict(),
      'opt_task': opt_task.state_dict(),
      'opt_pretext': opt_pretext.state_dict(),
    }, path)

  def load_networks(path):
    l = torch.load(path)
    model_rgb.load_state_dict(l['model_rgb'])
    model_d.load_state_dict(l['model_d'])
    model_task.load_state_dict(l['model_task'])
    model_pretext.load_state_dict(l['model_pretext'])

    opt_rgb.load_state_dict(l['opt_rgb'])
    opt_d.load_state_dict(l['opt_d'])
    opt_task.load_state_dict(l['opt_task'])
    opt_pretext.load_state_dict(l['opt_pretext'])





  # ======= TRAINING ========
  def train_model(model, batch, loss_weight=None):
    """
      Train either: model_task, model_pretext on a specific batch.
      Also perform backpropagation of gradients.
    """
    rgb, d, gt = batch
    rgb, d, gt = rgb.to(device), d.to(device), gt.to(device)

    f = combine_modes(rgb, d)
    pred = model(f)
    loss = criterion(pred, gt)

    if loss_weight is not None:
      loss *= loss_weight

    loss.backward()
    del rgb, d, gt, f, pred, loss


  def epoch_train():
    """
      Train a single epoch
    """
    print("\n====> TRAINING")

    for m in model_list:
      m.train()

    iter_source = LoaderIterator(dl_train_source, skip_last=True)
    iter_source_pt = LoaderIterator(dl_train_source_pt, skip_last=True)
    iter_target_pt = LoaderIterator(dl_train_target_pt, infinite=True, skip_last=True)
    iter_target = LoaderIterator(dl_train_target, infinite=True, skip_last=True)
    vs = [0, 0, 0, 0]
    # ITERATIONS
    for source_batch in tqdm(iter_source):      
      for o in opt_list: # zero all gradients
        o.zero_grad()

      # recognition on source
      rgb, d, gt = source_batch
      rgb, d, gt = rgb.to(device), d.to(device), gt.to(device)
      f = combine_modes(rgb, d)
      logits = model_task(f)
      loss_rec = criterion(logits, gt)

      loss_ent = 0

      # ent loss
      if hp.ent_weight > 0.:
        rgb, d, gt = next(iter_target)
        rgb, d, gt = rgb.to(device), d.to(device), gt.to(device)

        f = combine_modes(rgb, d)
        logits = model_task(f)
        loss_ent = ent_loss(logits)
      else:
        loss_ent = 0

      loss = loss_rec + hp.ent_weight * loss_ent
      loss.backward()
      # del rgb, d, loss

      

      if hp.pretext_weight > 0.:
        # rotation on source
        source_batch_pt = next(iter_source_pt)
        train_model(model_pretext, source_batch_pt, hp.pretext_weight)

        # rotation on target
        target_batch_pt = next(iter_target_pt)
        train_model(model_pretext, target_batch_pt, hp.pretext_weight)

        for l in target_batch_pt[2]:
          vs[l] += 1

      for o in opt_list: #update weights
        o.step()

    print(vs)


  # ========== EVALUATION ==============
  def eval_model(model, loader, desc="EVAL", limit_samples=None, tag="none", step=0):
    """
      Validate either: model_task, model_pretext on the dataset contained in loader.
      Report number of correct guesses and avg loss per batch.
      If limit_samples is not None only take first n samples
    """
    print(f"\n--> {desc}")

    loss = 0.0
    correct = 0.0
    total = 0

    num_batches = len(loader)

    if limit_samples:
      limit_batches = math.floor(limit_samples/loader.batch_size)
      num_batches = min(limit_batches, num_batches)

    # TEST SOURCE CLASSIFICATION PERFORMANCES
    with torch.no_grad():
      for i, (rgb, d, gt) in tqdm(enumerate(loader), total=num_batches):
        if i >= num_batches: # terminate if we have more batches than we want
          break

        rgb, d, gt = rgb.to(device), d.to(device), gt.to(device)

        f = combine_modes(rgb, d)
        pred = model(f)

        loss += criterion(pred, gt).item()
        correct += (torch.argmax(pred, dim=1)==gt).sum().item()
        total += pred.size(0)

    accuracy = (correct / total)*100

    print(f"\nRESULTS: {loss:.2f} | {accuracy:.1f}% ({int(correct)}/{total})\n")
    summary.add_scalar(f"accuracy/{tag}", accuracy, step)
    summary.add_scalar(f"loss/{tag}", loss, step)


  def epoch_eval(n):
    """
      Evaluate at each epoch
    """
    print("\n\n====> EVALUATING")

    for m in model_list:
      m.eval()

    eval_model(model_task, dl_eval_source, step=n, desc="SOURCE: CLASSIFICATION", tag="source_recog")

    if hp.pretext_weight > 0:
      eval_model(model_pretext, dl_eval_source_pt, step=n, desc="SOURCE: ROTATION", tag="source_rot")
      eval_model(model_pretext, dl_eval_target_pt, step=n, desc="TARGET: ROTATION", tag="target_rot", limit_samples=8000, )
    else:
      print("Skipping evaluation for pretext task since pretext_weight is 0...\n")

    eval_model(model_task, dl_eval_target, step=n, desc="TARGET: CLASSIFICATION", tag="target_recog", limit_samples=8000)


  # ===================================
  # =========== LOOP ==================
  # ===================================
  start_epoch = 1

  # models at each epoch are saved in "snapshots/{hyperparamers}" as "model_{epoch}"
  # where `hyperparamers` is a string that represents hps like: "epochs_2__batch_size_128__lr_0.0003__momentum_0.9__weight_decay_0__pretext_weight_1"
  # and {epoch} is the epoch at which the model was trained
  snapshot_folder = os.path.join('snapshots', hp.to_filename())
  os.makedirs(snapshot_folder, exist_ok=True)

  if resume:
    epochs = get_epochs_in_model_folder(snapshot_folder)
        
    if len(epochs) > 0:
      recent_epoch = max(epochs)
      recent_model = f"model_{recent_epoch}.tar"
      start_epoch = recent_epoch+1

      print(f"\n\n\n{'='*12} RESUME EPOCH {recent_epoch}/{hp.epochs} {'='*12}")

      load_networks(os.path.join(snapshot_folder, recent_model))
      print(f"--> Model loaded from file! Running evaluation for resumed model...")

      epoch_eval(n=recent_epoch)
  

  for e in range(start_epoch, hp.epochs+1):
    print(f"\n\n\n{'='*12} EPOCH {e}/{hp.epochs} {'='*12}")

    epoch_train()

    if save_snapshots:
      save_networks(os.path.join(snapshot_folder, f"model_{e}.tar"))

    epoch_eval(n=e)

  print("\n\n+++ COMPLETED! +++\n\n", "\n"*3)



# ====== FILE ENTRY POINT ======
if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  
  setup_hp_arguments(parser)
  parser.add_argument("--no-resume", action="store_true", help="Do not resume training. Start from epoch 1")
  args = parser.parse_args()

  # load hp from arguments
  hp = HP()
  hp.load_from_args(args)

  # run the script
  resume = not args.no_resume
  run(hp, resume=resume)
