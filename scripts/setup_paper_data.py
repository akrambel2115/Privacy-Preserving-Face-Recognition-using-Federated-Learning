import os
import shutil

source_dir = "dataset/webface_112x112"
target_dir = "dataset/paper_1000_clients"

os.makedirs(target_dir, exist_ok=True)
all_identities = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

# Take exactly 1,000 clients to match the paper
paper_subset = all_identities[:1000]

print(f"Copying 1,000 identities to {target_dir}...")
for identity in paper_subset:
    src = os.path.join(source_dir, identity)
    dst = os.path.join(target_dir, identity)
    if not os.path.exists(dst):
        shutil.copytree(src, dst)

print("Done! Dataset is ready for paper validation.")