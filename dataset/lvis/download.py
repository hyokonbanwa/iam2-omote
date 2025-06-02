from pathlib import Path
import os
from ultralytics.utils.downloads import download

# Download labels
dir = "/data_ssd/lvis"  # dataset root dir
url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
urls = [f"{url}lvis-labels-segments.zip"]
download(urls, dir=dir)

# Download data
urls = [
    "http://images.cocodataset.org/zips/train2017.zip",  # 19G, 118k images
    "http://images.cocodataset.org/zips/val2017.zip",  # 1G, 5k images
    "http://images.cocodataset.org/zips/test2017.zip",  # 7G, 41k images (optional)
]
download(urls, dir=os.path.join(dir,"images"), threads=3)
