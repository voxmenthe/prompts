import sys
from zipfile import ZipFile
from huggingface_hub import hf_hub_download

repo = "hf-doc-build/doc-build"
version = "v5.0.0rc1"

print(f"Downloading/Locating transformers/{version}.zip...")
try:
    path = hf_hub_download(repo, filename=f"transformers/{version}.zip", repo_type="dataset")
    print(f"Path: {path}")

    with ZipFile(path, "r") as zf:
        names = zf.namelist()
        print(f"Total items: {len(names)}")
        
        prefix = f"transformers/{version}/"
        non_lang_files = []
        
        for name in names:
            if not name.startswith(prefix):
                continue
            rel = name[len(prefix):]
            if not rel: continue
            
            parts = rel.split('/')
            if not parts: continue
            
            # Check if first part is a 2-letter code
            first = parts[0]
            if len(first) == 2 and (len(parts) > 1 or name.endswith('/')):
                 continue
                 
            non_lang_files.append(name)
            
        print(f"\nNon-lang files found ({len(non_lang_files)}):")
        for n in non_lang_files[:20]:
            print(f"  {n}")
            
except Exception as e:
    print(f"Failed to check zip: {e}")

print("\nDownloading _versions.yml...")
try:
    v_path = hf_hub_download(repo, filename="transformers/_versions.yml", repo_type="dataset")
    with open(v_path, 'r') as f:
        print(f.read())
except Exception as e:
    print(f"Failed to read versions: {e}")
