import os
import glob
import torch
import pandas as pd
import numpy as np
import sys

def process_directory(base_dir):
    headers = ['s_winsize_sum', 's_winsize_med', 's_winsize_mean', 's_winsize_std', 'd_winsize_mean', 'd_winsize_sum', 'd_winsize_med', 's_bytes_sum', 's_bytes_max', 'd_winsize_std', 's_bytes_std', 'd_bytes_sum', 's_bytes_mean', 'd_bytes_mean', 's_winsize_min', 'd_winsize_min', 'd_bytes_std', 'd_ttl_sum', 'd_winsize_max', 'dur', 'd_bytes_max', 's_iat_sum', 'd_iat_max', 's_winsize_max', 's_iat_max', 'd_iat_sum', 's_iat_mean', 'd_load', 'syn_ack', 's_load', 'd_iat_mean', 'd_ttl_mean', 's_iat_std', 'd_ttl_max', 'd_ttl_med', 'd_ttl_min', 'tcp_rtt', 'd_iat_med', 's_iat_med', 'd_bytes_med', 'd_iat_std', 's_ttl_sum', 'ack_cnt', 'd_iat_min', 'd_port', 's_port', 's_pkt_cnt', 'd_pkt_cnt', 's_bytes_med', 'ack_dat', 'psh_cnt', 's_iat_min', 's_bytes_min', 's_ttl_min', 's_ttl_max', 's_ttl_med', 'd_bytes_min', 's_ttl_mean', 'syn_cnt', 'fin_cnt', 'rst_cnt', 'd_ttl_std', 'ece_cnt', 'cwr_cnt', 'proto', 'pkt_depth', 'neg_f1_score', 'compute_cost', 'Timestamp']
    
    # Construct search pattern for the given directory
    search_pattern = os.path.join(base_dir, '**/*.pt')
    pt_files = glob.glob(search_pattern, recursive=True)
    
    if not pt_files:
        print(f"No .pt files found in {base_dir} or its subdirectories.")
        return

    for i, file in enumerate(pt_files):
        try:
            data = torch.load(file, map_location=torch.device('cpu'), weights_only=False)
            X, Y = data.get('X_history'), data.get('Y_history')
            
            if X is None or Y is None: 
                print(f"Skipping {file}: Missing X_history or Y_history")
                continue
                
            X_np, Y_np = X.numpy(), Y.numpy()
            combined = np.hstack((X_np, Y_np))
            
            # 1. Create DF with initial mapping
            if combined.shape[1] == len(headers) - 1:
                df = pd.DataFrame(combined, columns=headers[:-1])
                df['Timestamp'] = 0 
            else:
                df = pd.DataFrame(combined, columns=headers[:combined.shape[1]])

            # 2. CRITICAL FIX: Direct Sign Correction
            if 'neg_f1_score' in df.columns:
                # Convert positive F1 (0.9) to negative F1 (-0.9)
                df['neg_f1_score'] = -df['neg_f1_score'].abs()
                
            if 'compute_cost' in df.columns:
                # Convert negative cost (-3000) to positive raw cost (3000)
                df['compute_cost'] = df['compute_cost'].abs()

            df['source_experiment'] = os.path.dirname(file)

            # 3. Clean up Booleans
            for col in df.columns:
                if col not in ['neg_f1_score', 'compute_cost', 'Timestamp', 'source_experiment', 'pkt_depth']:
                    df[col] = df[col].map({1.0: True, 0.0: False, 1: True, 0: False}).astype(bool)

            # 4. Generate unique filename and save in the same directory
            dir_name = os.path.dirname(file)
            base_name = os.path.splitext(os.path.basename(file))[0]
            
            # Construct output filename with _i appended
            output_filename = os.path.join(dir_name, f"{base_name}_{i}.csv")
            
            df.to_csv(output_filename, index=False)
            print(f"Saved {len(df)} points from {file} into {output_filename}")
            
        except Exception as e:
            print(f"Failed to process {file}. Error: {e}")

if __name__ == "__main__":
    # Ensure a directory path is passed as an argument
    if len(sys.argv) < 2:
        print("Usage: python pt_to_csv.py <directory_path>")
        sys.exit(1)
        
    target_dir = sys.argv[1]
    
    if not os.path.isdir(target_dir):
        print(f"Error: '{target_dir}' is not a valid directory.")
        sys.exit(1)
        
    process_directory(target_dir)