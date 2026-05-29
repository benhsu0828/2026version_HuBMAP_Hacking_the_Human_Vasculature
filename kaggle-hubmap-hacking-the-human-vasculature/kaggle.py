import kagglehub

# Download latest version
path = kagglehub.dataset_download("tascj0/hubmap-2023-configs")

print("Path to dataset files:", path)