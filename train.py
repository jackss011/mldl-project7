import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from tqdm import tqdm
import math
from dataclasses import dataclass

from datasets import PretextSynRODDataset, PretextRODDataset, RODDataset, SynRODDataset
from networks import FeatureExtractor, RecognitionClassifier, RotationClassifier, weight_init
from utils import LoaderIterator, show_image


TRAIN_WORKERS = 2   # workers used for loading training data (for each dataset)
EVAL_WORKERS = 4    # workers used for loading evaluation data (for each dataset)


# HYPERPARAMS
@dataclass
class HP:
  epochs = 2
  batch_size = 64*2
  lr = 3e-4
  momentum = 0.9
  weight_decay = 0
  pretext_weight = 1


def run(hp: HP):
  """
    Performs a full run with the specified hyperparameters
  """
  # select device
  device = None
  if torch.cuda.is_available():
    sel_dev = torch.cuda.current_device()
    print(f"+++ DEVICE INFO: {torch.cuda.device_count()} cuda devices available. Using {torch.cuda.get_device_name(sel_dev)}")
    device = 'cuda'
  else:
    print("No CUDA device available. Using CPU")
    device = 'cpu'


  # ======= DATASETS ========
  ds_train_source = SynRODDataset("data", train=True, image_size=224)             # Source labelled
  ds_train_source_pt = PretextSynRODDataset("data", train=True, image_size=224)   # Source pretext
  ds_train_target_pt = PretextRODDataset("data", train=True, image_size=224)      # Target pretext

  ds_eval_source = SynRODDataset("data", train=False, image_size=224)             # Test if can classify source
  ds_eval_source_pt = PretextSynRODDataset("data", train=False, image_size=224)   # Test if can predict rotation of source images
  ds_eval_target_pt = PretextRODDataset("data", train=True, image_size=224)       # Test if can predict rotation of target images


  # ======= DATALOADERS ========
  def dataloader_factory(ds, num_workers=TRAIN_WORKERS):
    return DataLoader(ds, batch_size=hp.batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)

  dl_train_source = dataloader_factory(ds_train_source)
  dl_train_source_pt = dataloader_factory(ds_train_source_pt)
  dl_train_target_pt = dataloader_factory(ds_train_target_pt)

  dl_eval_source = dataloader_factory(ds_eval_source, num_workers=EVAL_WORKERS)
  dl_eval_source_pt = dataloader_factory(ds_eval_source_pt, num_workers=EVAL_WORKERS)
  dl_eval_target_pt = dataloader_factory(ds_eval_target_pt, num_workers=EVAL_WORKERS)


  # ======= MODELS ========
  model_rgb = FeatureExtractor().to(device)
  model_d = FeatureExtractor().to(device)

  def combine_modes(rgb, d) -> torch.Tensor:
    return torch.cat((model_rgb(rgb), model_d(d)), dim=1)

  model_task = RecognitionClassifier(512*2, 51).to(device)
  model_pretext = RotationClassifier(512*2, 4).to(device)

  model_task.apply(weight_init)
  model_pretext.apply(weight_init)

  criterion = torch.nn.CrossEntropyLoss().to(device)


  # ======= OPTIMIZERS ========
  def opt_factory(model):
    return SGD(model.parameters(), lr=hp.lr, momentum=hp.momentum, weight_decay=hp.weight_decay)

  opt_rgb = opt_factory(model_rgb)
  opt_d = opt_factory(model_d)
  opt_task = opt_factory(model_task)
  opt_pretext = opt_factory(model_pretext)

  opt_list = [opt_rgb, opt_d, opt_task, opt_pretext]



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

    iter_source = LoaderIterator(dl_train_source, skip_last=True)
    iter_source_pt = LoaderIterator(dl_train_source_pt, skip_last=True)
    iter_target_pt = LoaderIterator(dl_train_target_pt, infinite=True, skip_last=True)

    # ITERATIONS
    for source_batch in tqdm(iter_source):      
      for o in opt_list: # zero all gradients
        o.zero_grad()

      # recognition on source
      train_model(model_task, source_batch)

      if hp.pretext_weight > 0:
        # rotation on source
        source_batch_pt = next(iter_source_pt)
        train_model(model_pretext, source_batch_pt, hp.pretext_weight)

        # rotation on target
        target_batch_pt = next(iter_target_pt)
        train_model(model_pretext, target_batch_pt, hp.pretext_weight)

      for o in opt_list: #update weights
        o.step()


  # ========== EVALUATION ==============
  def eval_model(model, loader, desc="EVAL", limit_samples=None):
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

    accuracy = correct / total
    loss_per_batch = loss / num_batches
    print(f"\nRESULTS: {loss_per_batch:.2f} | {accuracy*100:.1f}% ({int(correct)}/{total})\n")


  def epoch_eval():
    """
      Evaluate at each epoch
    """
    print("\n\n====> EVALUATING")

    eval_model(model_task, dl_eval_source, desc="SOURCE: CLASSIFICATION")

    if hp.pretext_weight > 0:
      eval_model(model_pretext, dl_eval_source_pt, desc="SOURCE: ROTATION")
      eval_model(model_pretext, dl_eval_target_pt, desc="TARGET: ROTATION", limit_samples=8000)
    else:
      print("Skipping evaluation for pretext task since pretext_weight is 0...")


  # ===================================
  # =========== LOOP ==================
  # ===================================
  for e in range(1, hp.epochs+1):
    print(f"\n\n\n{'='*12} EPOCH {e}/{hp.epochs} {'='*12}")
    epoch_train()
    epoch_eval()

  print("\n\n+++ COMPLETED! +++\n\n", "\n"*3)



# run with basic hp for now
if __name__ == '__main__':
    run(HP())
