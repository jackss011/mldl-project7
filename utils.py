import os
from tenacity import retry
import torchvision.transforms.functional as TF

def download_file(url, filename):
  """Download the file at @url and save with @filename"""
  import requests

  if os.path.exists(filename):
    raise FileExistsError(filename)
  
  os.makedirs(os.path.dirname(filename), exist_ok=True)

  with requests.get(url, stream=True) as r:
    r.raise_for_status()

    file_size = r.headers['Content-Length']
    print(f'Downloading file ({round(int(file_size)/2**30, 2)}GB)')
    print("!!! If interrupted delete partially downloaded file manually!!!")
    
    with open(filename, 'wb+') as f:
      downloaded_size = 0
      chunk_size = 8192

      for chunk in r.iter_content(chunk_size=chunk_size): 
        f.write(chunk)
        downloaded_size += chunk_size

    print('Download Complete!')
  return True


def extract_tar(tar_path, dest_path):
  """Extract a tar file into the specified path"""
  import tarfile

  with tarfile.open(tar_path, 'r') as t:
    print('Extracting...')
    t.extractall(path=dest_path)
    print('Extracting Done!')


def rotate_image(img, r):
  """
    Args:
      img (Tensor): (..., H, W) Tensor image
      r (int): rotate by 90*r degrees
  """
  if r == 0:
    return img
  if r == 1:
    return TF.hflip(img.moveaxis(2, 1))
  if r == 2:
    return TF.hflip(TF.vflip(img))
  if r == 3:
    return TF.vflip(img.moveaxis(2, 1))

  raise ValueError(f"Value of r in in image rotation is invalid: {r}")


from matplotlib import pyplot as plt
def show_image(t, *, ax=plt, editor=False):
  import numpy as np
  img = t.numpy()
  img = np.moveaxis(img, 0, 2)
  ax.imshow(img)

  if editor:
    plt.show()


class LoaderIterator:
  def __init__(self, loader, *, skip_last=False, infinite=False) -> None:
    self.loader = loader
    self.skip_last = skip_last
    self.infinite = infinite

    self._iter = iter(self.loader)


  def __iter__(self):
    return self

  def __len__(self):
    if not self.infinite:
      return len(self.loader)
    else:
      return None
  

  def __next__(self):
    n = next(self._iter, None)

    if n is None or ( self.skip_last and (n[0].size(0) != self.loader.batch_size) ):
      if self.infinite:
        self._iter = iter(self.loader)
        n = next(self._iter)
      else:
        raise StopIteration()
    
    return n
