import kagglehub

# Download the pre-aligned 112x112 CASIA-WebFace dataset
path = kagglehub.dataset_download("yakhyokhuja/webface-112x112",output_dir="dataset/")

print("Path to dataset files:", path)