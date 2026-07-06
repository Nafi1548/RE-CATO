import torch
import matplotlib.pyplot as plt

# 1. Define the paths to your results
path_merged = '/home/nafi/CATO/morbo_copy/experiments/rover/results_seed_1.pt'
path_morbo = '/home/nafi/CATO/morbo/experiments/rover/results_seed_1.pt'

# 2. Load the PyTorch dictionaries
# Use map_location='cpu' in case they were saved on a GPU but you are analyzing on a CPU
res_morbo = torch.load(path_morbo, map_location=torch.device('cpu'))
res_merged = torch.load(path_merged, map_location=torch.device('cpu'))

# 3. Extract the evaluation counts and hypervolume metrics
evals_morbo = res_morbo['n_evals']
hv_morbo = res_morbo['true_hv']

evals_merged = res_merged['n_evals']
hv_merged = res_merged['true_hv']

# 4. Plot the comparison
plt.figure(figsize=(10, 6))

# Plot baseline
plt.plot(evals_morbo, hv_morbo, label='Baseline MORBO', 
         color='blue', marker='o', markersize=4, alpha=0.8)

# Plot your merged framework
plt.plot(evals_merged, hv_merged, label='MORBO + CASMOPOLITAN', 
         color='orange', marker='s', markersize=4, alpha=0.8)

# Formatting the plot
plt.title('Rover Problem: Hypervolume vs. Number of Evaluations')
plt.xlabel('Number of Evaluations')
plt.ylabel('Hypervolume')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()