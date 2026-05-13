import kagglehub

# Download latest version
path = kagglehub.dataset_download("jessicali9530/lfw-dataset",output_dir="eval/")

print("Path to dataset files:", path)