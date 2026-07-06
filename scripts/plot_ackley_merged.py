import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Plot MORBO+Casmopolitan Ackley53 Performance.")
    parser.add_argument('--csv_path', type=str, default='./output/morbo_casmo_func2C_seed42.csv', 
                        help='Path to the merged MORBO+Casmopolitan CSV.')
    parser.add_argument('--save_plot', type=str, default='./output/merged_func2C_performance.png', 
                        help='Path to save the generated plot.')
    args = parser.parse_args()

    # 1. Check if the file exists
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found at {args.csv_path}")
        return

    # 2. Load the data
    print(f"Loading data from {args.csv_path}...")
    df = pd.read_csv(args.csv_path)

    # 3. Create the plot
    plt.figure(figsize=(10, 6))

    # Plot BestValue against Index (Iterations) using step-post
    plt.step(df['Index'], df['BestValue'], where='post', 
             label='MORBO + Casmopolitan (Merged)', color='red', linewidth=2.5, alpha=0.9)

    # 4. Formatting the chart
    plt.title('Optimization Performance: Ackley53 (Mixed Space)', fontsize=15, fontweight='bold')
    plt.xlabel('Number of Function Evaluations', fontsize=12)
    plt.ylabel('Best Objective Value Found (Minimization)', fontsize=12)
    
    # Ackley's global minimum is 0, so a logarithmic scale is best for viewing convergence
    plt.yscale('log') 
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # 5. Save and Show
    os.makedirs(os.path.dirname(args.save_plot), exist_ok=True)
    plt.savefig(args.save_plot, dpi=300)
    print(f"Plot successfully saved to: {args.save_plot}")
    
    try:
        plt.show()
    except Exception:
        print("Could not display plot interactively (check your display environment).")

if __name__ == '__main__':
    main()