import torch


print("PyTorch Version:", torch.__version__)

print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():

    print("CUDA Version:", torch.version.cuda)

    print("Number of GPUs:", torch.cuda.device_count())

    print("GPU Name:", torch.cuda.get_device_name(0))
