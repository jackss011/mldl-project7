import os

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
