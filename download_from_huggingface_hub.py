import huggingface_hub

import sys

arguments = sys.argv[1:]
huggingface_id = arguments[0]
local_dir = arguments[1]
if len(arguments) > 2:
    repo_type = arguments[2]
else:
    repo_type = None

download_flag = False

while not download_flag:
    try:
        huggingface_hub.snapshot_download(huggingface_id, local_dir=local_dir, local_dir_use_symlinks=False,repo_type=repo_type)
        download_flag = True
    except Exception as e:
        print(e)
        print("Download failed, retrying...")
        download_flag = False
        

