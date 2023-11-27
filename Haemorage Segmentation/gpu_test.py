import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))
