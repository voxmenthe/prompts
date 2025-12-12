from huggingface_hub import list_repo_files

repo = "hf-doc-build/doc-build"
print(f"Listing files in {repo}...")
files = list_repo_files(repo, repo_type="dataset")

print("Files found (first 100):")
for f in files[:100]:
    print(f)

print("\nSearching for 'transformers' zip files...")
for f in files:
    if "transformers" in f and f.endswith(".zip"):
        print(f"  {f}")
