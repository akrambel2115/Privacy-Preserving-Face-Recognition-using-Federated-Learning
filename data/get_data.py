import kagglehub

# Download latest version
path = kagglehub.dataset_download("vishesh1412/celebrity-face-image-dataset",output_dir="celebs/")

print("Path to dataset files:",path)