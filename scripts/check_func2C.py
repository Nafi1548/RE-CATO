
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot and compare Func2C BO performance.")
    parser.add_argument('--baseline', type=str, default='/home/nafi/CATO/casmopolitan/baseline_func2C.csv', 
                        help='Path to the Casmopolitan baseline CSV.')
    parser.add_argument('--merged', type=str, default='./output/morbo_casmo_func2C_seed42.csv', 
                        help='Path to the merged MORBO+Casmopolitan CSV.')
    parser.add_argument('--save_plot', type=str, default='./output/func2C_comparison.png', 
                        help='Path to save the generated plot.')
    args = parser.parse_args()

    # 1. Check if files exist
    if not os.path.exists(args.baseline):
        print(f"Error: Baseline file not found at {args.baseline}")
        return
    if not os.path.exists(args.merged):
        print(f"Error: Merged framework file not found at {args.merged}")
        return

    # 2. Load the data
    print("Loading data...")
    df_baseline = pd.read_csv(args.baseline)
    df_merged = pd.read_csv(args.merged)

    # 3. Create the plot
    plt.figure(figsize=(10, 6))

    # Plot BestValue against Index (Iterations)
    # Using step-post to clearly show when the incumbent best value drops
    plt.step(df_baseline['Index'], df_baseline['BestValue'], where='post', 
             label='Casmopolitan (Baseline)', color='blue', linewidth=2, alpha=0.8)
    
    plt.step(df_merged['Index'], df_merged['BestValue'], where='post', 
             label='MORBO + Casmo (Merged)', color='red', linewidth=2, alpha=0.8)

    # 4. Formatting the chart
    plt.title('Optimization Performance: Func2C (Mixed Space)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Function Evaluations', fontsize=12)
    plt.ylabel('Best Objective Value Found (Minimization)', fontsize=12)
    
    # Note: Func2C has a global minimum around -0.2063. 
    # Do NOT use a log scale here since values drop below zero.
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    # Add a horizontal line showing the true global minimum for reference
    plt.axhline(y=-0.2063, color='green', linestyle=':', linewidth=2, label='Global Minimum (-0.2063)')
    
    plt.legend(fontsize=12)
    plt.tight_layout()

    # 5. Save and Show
    os.makedirs(os.path.dirname(args.save_plot), exist_ok=True)
    plt.savefig(args.save_plot, dpi=300)
    print(f"Plot successfully saved to: {args.save_plot}")
    
    # Try to display if the environment supports it
    try:
        plt.show()
    except Exception as e:
        print("Could not display plot interactively (likely running in a headless environment).")

if __name__ == '__main__':
    main()