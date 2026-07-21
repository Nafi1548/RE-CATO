import pandas as pd
import argparse
import sys
from pathlib import Path

def main():
    # 1. Set up argument parsing for just the input path
    parser = argparse.ArgumentParser(description="Transform specific columns and auto-generate output file.")
    parser.add_argument("input_path", help="The path to the source CSV file (e.g., my_data.csv)")
    args = parser.parse_args()

    # 2. Safely generate the new output path using pathlib
    input_file = Path(args.input_path)
    
    # This creates a new filename like: "converted_casmo_global_rf_my_data.csv"
    # and keeps it in the same directory as the input file.
    new_filename = f"converted_casmo_global_rf_{input_file.name}"
    output_path = input_file.with_name(new_filename)

    try:
        # 3. Read the actual CSV file
        df = pd.read_csv(input_file)

        # 4. Transform the data
        df['pkt_depth'] = df['Packet_Depth']
        df['neg_f1_score'] = -df['F1_Score']
        df['compute_cost'] = df['Compute_Cost']

        # Select only the new columns in the requested order
        new_df = df[['pkt_depth', 'neg_f1_score', 'compute_cost']]

        # 5. Save to the auto-generated output path
        new_df.to_csv(output_path, index=False)
        print(f"Success!")
        print(f"Input:  {input_file}")
        print(f"Output: {output_path}")

    except FileNotFoundError:
        print(f"Error: The input file '{input_file}' was not found.")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: The input file is missing a required column: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()