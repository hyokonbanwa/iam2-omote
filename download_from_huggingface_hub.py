import huggingface_hub

import sys

from huggingface_hub import get_token, HfApi

def hf_login_status():
    token = get_token()  # 環境変数やローカルキャッシュからトークンを取得（なければ None）
    if not token:
        return {"logged_in": False, "reason": "No token configured."}

    try:
        info = HfApi().whoami(token=token)  # トークンの有効性＆ユーザー情報を確認
        return {
            "logged_in": True,
            "user": info.get("name"),
            "email": info.get("email"),
            "orgs": [o.get("name") for o in info.get("orgs", [])],
        }
    except Exception as e:
        return {"logged_in": False, "reason": f"Token found but invalid or not usable: {e}"}



arguments = sys.argv[1:]
huggingface_id = arguments[0]
local_dir = arguments[1]
if len(arguments) > 2:
    repo_type = arguments[2]
else:
    repo_type = None

download_flag = False

login_status = hf_login_status()
print("Login status:", login_status)
if not login_status['logged_in']:
    print("You are not logged in to Hugging Face Hub. Please log in first.")
    sys.exit(1)
    
while not download_flag:
    try:
        huggingface_hub.snapshot_download(huggingface_id, local_dir=local_dir, local_dir_use_symlinks=False,repo_type=repo_type)
        download_flag = True
    except Exception as e:
        print(e)
        print("Download failed, retrying...")
        download_flag = False
        

