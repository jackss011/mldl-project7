from os import path
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor
from utils import download_file, extract_tar

DATASETS_URL = 'https://www.dropbox.com/s/xdy5cfu7m63pk46/?dl=1'
DATASETS_NAME = 'ROD-synROD'


def ensure_download(root):
  """
    Check if the datasets is downloaded in the @root folder. Donwload and extract it otherwise.
  """
  tar_filename = path.join(root, DATASETS_NAME + '.tar')
  datasets_folder = path.join(root, DATASETS_NAME)

  has_tarfile = path.exists(tar_filename)
  has_extracted_folder = path.exists(datasets_folder)

  if not has_tarfile and not has_extracted_folder:
    download_file(DATASETS_URL, tar_filename)

  if not has_extracted_folder:
    extract_tar(tar_filename, root)



class BaseDataset(Dataset):
  """
    Base class for the datasets. Both ROD and SynROD are similar in structure and this class will reuse code for them.
    The main difference is when mapping the image path in the annotaions to the real image path.
    The _map_image_path() function is extended in sublasses to retrieve depth and rgb paths.
    Example for ROD, if we have path in annotations: rubber_eraser/rubber_eraser_4/rubber_eraser_4_1_136_***.png
    Then it will mapped to the following to retrieve images:
      - for rgb: rgb-washington/rubber_eraser/rubber_eraser_4/rubber_eraser_4_1_136_crop.png
      - for depth: surfnorm-washington/rubber_eraser/rubber_eraser_4/rubber_eraser_4_1_136_depthcrop.png
  """
  def __init__(self, subfolder, annotations_filename, root, download=False, image_size=None, transform=ToTensor()):
    self.root = root
    self.download = download
    self.image_size = image_size
    self.transform = transform

    if download:
      ensure_download(root)

    self.images_folder = path.join(root, DATASETS_NAME, subfolder)
    annotations_path = path.join(self.images_folder, annotations_filename)

    with open(annotations_path, 'r') as f:
      self.annotations = [l.split(' ') for l in f.readlines()]
      self.annotations = [(p, int(l)) for p, l in self.annotations]

  def __len__(self):
    return len(self.annotations)

  def __getitem__(self, idx):
    img_path, label = self.annotations[idx]

    rgb_image_path = path.join(self.images_folder, self._map_image_path(img_path, is_rgb=True))
    d_image_path = path.join(self.images_folder, self._map_image_path(img_path, is_rgb=False))

    rgb_image = Image.open(rgb_image_path)
    d_image = Image.open(d_image_path)

    if self.image_size:
      s = (self.image_size, self.image_size)
      rgb_image = rgb_image.resize(s)
      d_image = d_image.resize(s)
    
    if self.transform:
      rgb_image = self.transform(rgb_image)
      d_image = self.transform(d_image)

    return rgb_image, d_image, label

  def _map_image_path(self, path, is_rgb):
    pass



class RODDataset(BaseDataset):
  """ROD datasets. Made of couples of RGB and Depth pictures from the real world. Target domain"""
  def __init__(self, root, download=False, image_size=None):
      super().__init__('ROD', 'wrgbd_40k-split_sync.txt', root, download, image_size)

  def _map_image_path(self, img_path, is_rgb):
    if is_rgb:
      return path.join('rgb-washington', img_path.replace('***', 'crop'))
    else:
      return path.join('surfnorm-washington', img_path.replace('***', 'depthcrop'))



class SynRODDataset(BaseDataset):
  """ROD datasets. Made of couples of RGB and Depth pictures from a 3D rendered. Source domain"""
  def __init__(self, root, download=False, image_size=None, train=True):
      self.train = train

      annotations = None
      if train:
        annotations = 'synARID_50k-split_sync_train1.txt'
      else:
        annotations = 'synARID_50k-split_sync_test1.txt'
      
      super().__init__('synROD', annotations, root, download, image_size)

  def _map_image_path(self, img_path, is_rgb):
    if is_rgb:
      return img_path.replace('***', 'rgb')
    else:
      return img_path.replace('***', 'depth')
 