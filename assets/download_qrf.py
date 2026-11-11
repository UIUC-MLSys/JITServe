import os
from huggingface_hub import snapshot_download

def download_predictor():
    repo_id = "En-2863/jitserve-qrf-length-predictor"
    local_dir = "assets/qrf/"
    
    print(f"🚀 Starting download of {repo_id} to {local_dir}...")
    
    try:
        # snapshot_download is the underlying function for the CLI download
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            # In newer versions, local_dir implies local_dir_use_symlinks=False,
            # but we'll set it explicitly for clarity/compatibility.
            local_dir_use_symlinks=False,
            # Enable multi-threaded downloading for speed
            max_workers=8
        )
        
        print(f"✅ Download successful! Files are located in: {path}")
        
        # Verify the contents
        print("📂 Directory contents:")
        for file in os.listdir(local_dir):
            print(f" - {file}")
            
    except Exception as e:
        print(f"❌ Error during download: {e}")

if __name__ == "__main__":
    download_predictor()
