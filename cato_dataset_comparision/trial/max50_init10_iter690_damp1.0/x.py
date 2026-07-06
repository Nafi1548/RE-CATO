import torch

# Load the .pt file directly
data = torch.load('bo_results.pt', map_location='cpu')

print(f"Data type: {type(data)}")

if isinstance(data, dict):
    print("\nKeys found in the file:")
    for key, value in data.items():
        if torch.is_tensor(value):
            print(f" - '{key}': Tensor of shape {value.shape}")
        elif isinstance(value, list):
            print(f" - '{key}': List of length {len(value)}")
        else:
            print(f" - '{key}': {type(value)}")