import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from tqdm import tqdm
# import time, math

from datasets import PretextSynRODDataset, PretextRODDataset, RODDataset, SynRODDataset
from networks import FeatureExtractor, RecognitionClassifier, RotationClassifier
from utils import LoaderIterator, show_image



# HYPERPARAMS
EPOCHS = 2
BATCH_SIZE = 64*2
LR = 3e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 0

# CONFIG
NUM_WORKERS = 2


def main():
  # SELECT DEVICE
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
  ds_eval_source_pt = PretextSynRODDataset("data", train=False, image_size=224) 
  ds_eval_target_pt = PretextRODDataset("data", train=True, image_size=224) 


  # ======= DATALOADERS ========
  def dataloader_factory(ds, num_workers=NUM_WORKERS):
    return DataLoader(ds, batch_size=BATCH_SIZE, num_workers=num_workers, shuffle=True, pin_memory=True)

  dl_train_source = dataloader_factory(ds_train_source)
  dl_train_source_pt = dataloader_factory(ds_train_source_pt)
  dl_train_target_pt = dataloader_factory(ds_train_target_pt)

  dl_eval_source = dataloader_factory(ds_eval_source, num_workers=8)
  dl_eval_source_pt = dataloader_factory(ds_eval_source_pt, num_workers=8)
  dl_eval_target_pt = dataloader_factory(ds_eval_target_pt, num_workers=8)

  # ======= MODELS ========
  model_rgb = FeatureExtractor().to(device)
  model_d = FeatureExtractor().to(device)

  def combine_modes(rgb, d) -> torch.Tensor:
    return torch.cat((model_rgb(rgb), model_d(d)), dim=1)

  model_task = RecognitionClassifier(512*2, 51).to(device)
  model_pretext = RotationClassifier(512*2, 4).to(device)

  criterion = torch.nn.CrossEntropyLoss().to(device)


  # ======= OPTIMIZERS ========
  def opt_factory(model):
    return SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

  opt_rgb = opt_factory(model_rgb)
  opt_d = opt_factory(model_d)
  opt_task = opt_factory(model_task)
  opt_pretext = opt_factory(model_pretext)

  opt_list = [opt_rgb, opt_d, opt_task, opt_pretext]



  # ======= TRAINING LOOP ========

  # ==================================
  def epoch_train():
    """
      Train a single epoch
    """
    print("\n====> TRAINING")

    def train_model(model, batch):
      rgb, d, gt = batch
      rgb, d, gt = rgb.to(device), d.to(device), gt.to(device)

      f = combine_modes(rgb, d)
      pred = model(f)
      loss = criterion(pred, gt)
      loss.backward()
      del rgb, d, gt, f, pred, loss

    iter_source = LoaderIterator(dl_train_source, skip_last=True)
    iter_source_pt = LoaderIterator(dl_train_source_pt, skip_last=True)
    iter_target_pt = LoaderIterator(dl_train_target_pt, infinite=True, skip_last=True)

    # ITERATIONS
    for source_batch in tqdm(iter_source):      
      # zero all gradients
      for o in opt_list:
        o.zero_grad()

      # +++ TASK +++
      train_model(model_task, source_batch)

      # +++ PRETEXT SOURCE +++
      source_batch_pt = next(iter_source_pt)
      train_model(model_pretext, source_batch_pt)

      # +++ PRETEXT TARGET +++
      target_batch_pt = next(iter_target_pt)
      train_model(model_pretext, target_batch_pt)

      for o in opt_list:
        o.step()


  # ==================================
  def epoch_eval():
    """
      Evaluate at each epoch
    """
    print("\n====> EVALUATING")

    def eval_model(model, loader, desc="EVAL"):
      print(f"\n--> {desc}")

      loss = 0.0
      correct = 0.0
      total = 0

      # TEST SOURCE CLASSIFICATION PERFORMANCES
      with torch.no_grad():
        for rgb, d, gt in tqdm(loader):
          rgb, d, gt = rgb.to(device), d.to(device), gt.to(device)

          f = combine_modes(rgb, d)
          pred = model(f)

          loss += criterion(pred, gt).item()
          correct += (torch.argmax(pred, dim=1)==gt).sum().item()
          total += pred.size(0)

      accuracy = correct / total
      loss_per_batch = loss / len(loader)
      print(f"\nRESULTS: {loss_per_batch:.2f} | {accuracy*100:.1f}% ({int(correct)}/{total})\n")

    eval_model(model_task, dl_eval_source, desc="SOURCE CLASSIFICATION")


  # =========== LOOP ==================
  for n in range(1, EPOCHS+1):
    print(f"\n\n\n{'='*12} EPOCH {n}/{EPOCHS} {'='*12}")
    epoch_train()
    epoch_eval()

  print("\n\n+++COMPLETED+++\n\n")



# call main function
if __name__ == '__main__':
    main()




# def train_model(model, batch):
#   rgb, d, gt = batch
#   rgb, d, gt = rgb.to(device), d.to(device), gt.to(device)

#   f = combine_modes(rgb, d)
#   pred = model(f)
#   loss = criterion(pred, gt)
#   loss.backward()
#   del rgb, d, gt, f, pred, loss