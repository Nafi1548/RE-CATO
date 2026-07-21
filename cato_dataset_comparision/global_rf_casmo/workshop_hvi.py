import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_pareto_front(data):
    """Extracts the non-dominated Pareto frontier from a given set of points."""
    is_pareto = np.ones(data.shape[0], dtype=bool)
    for i, c in enumerate(data):
        if is_pareto[i]:
            # Find points strictly dominated by c
            dominated = np.all(data >= c, axis=1) & np.any(data > c, axis=1)
            is_pareto[dominated] = False
    return data[is_pareto]

def calculate_2d_hypervolume(front, ref_point):
    """
    Calculates exact 2D hypervolume for minimization problems.
    Assumes front is a 2D numpy array of non-dominated points.
    """
    if len(front) == 0:
        return 0.0
    
    # Sort the Pareto front primarily by the first objective (ascending)
    front = front[np.argsort(front[:, 0])]
    
    hv = 0.0
    for i in range(len(front)):
        # Width is from current point's obj1 to the reference point's obj1
        width = ref_point[0] - front[i, 0]
        
        # Height is bounded by the previous point's obj2, or the reference point's obj2 for the first point
        if i == 0:
            height = ref_point[1] - front[i, 1]
        else:
            height = front[i-1, 1] - front[i, 1]
            
        # Ensure we don't add negative volume if a point exceeds the reference point
        if width > 0 and height > 0:
            hv += width * height
            
    return hv

def calculate_hvi_per_iteration(df, objectives, ref_point, initial_doe_size=10):
    """Calculates cumulative HVI over sequential iterations."""
    data = df[objectives].dropna().values
    hvi_history = []
    
    # Iterate through the dataset cumulatively
    for i in range(len(data)):
        # We look at all points evaluated up to iteration i
        current_data = data[:i+1]
        
        # Extract the current Pareto front
        current_front = get_pareto_front(current_data)
        
        # Calculate HVI for this front
        hv = calculate_2d_hypervolume(current_front, ref_point)
        hvi_history.append(hv)
        
    return hvi_history

# ==========================================
# Configuration
# ==========================================
# The objectives to minimize: Security risk (neg_f1_score) and processing latency (compute_time)
objectives = ['neg_f1_score', 'compute_cost']

baseline_csv = 'rubayet_baseline.csv'
approach_csv = 'rubayet_rf_mean_gp_var_approach1.csv'

df_base = pd.read_csv(baseline_csv)
df_app = pd.read_csv(approach_csv)
df_base = df_base[0:300]
df_app = df_app[0:300]
# Define the Reference Point (Worst Case Bounds)
# We find the global maximum of both objectives across BOTH datasets to ensure a fair comparison space.
max_f1 = max(df_base[objectives[0]].max(), df_app[objectives[0]].max())
max_compute = max(df_base[objectives[1]].max(), df_app[objectives[1]].max())

# Add a slight offset (e.g., 5%) so boundary points still contribute to the volume
ref_point = [max_f1 + abs(max_f1 * 0.05), max_compute + abs(max_compute * 0.05)]
print(f"Using Reference Point: {ref_point}")

# Calculate Histories
hvi_base = calculate_hvi_per_iteration(df_base, objectives, ref_point)
hvi_app = calculate_hvi_per_iteration(df_app, objectives, ref_point)

# ==========================================
# Plotting
# ==========================================
plt.figure(figsize=(10, 6))

# Plot lines
plt.plot(hvi_base, label='Baseline', color='#d62728', linewidth=2)
plt.plot(hvi_app, label='RF Mean & GP Var (Approach 1)', color='#1f77b4', linewidth=2.5)

# Formatting for thesis inclusion
plt.title('Hypervolume Indicator (HVI) Progression', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Number of Evaluated Samples (Iterations)', fontsize=13)
plt.ylabel('Hypervolume (Higher is Better)', fontsize=13)
plt.legend(fontsize=12, loc='lower right')
plt.grid(True, linestyle=':', alpha=0.7)
plt.xlim(0, max(len(hvi_base), len(hvi_app)))

plt.tight_layout()
plt.savefig('hvi_progression_local.png', dpi=300)
plt.show()