import kagglehub

# Download latest version
path = kagglehub.dataset_download("vishesh1412/celebrity-face-image-dataset",output_dir="data/celebs")

print("Path to dataset files:",path)