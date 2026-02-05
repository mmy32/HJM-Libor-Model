import pandas_datareader.data as web
import datetime
import pandas as pd
import os
import time

def get_treasury_matrix(start_date="2018-01-01"):
    # Mapping FRED ID to Tenor in Years
    symbol_map = {
        'DGS1MO': 0.0833, 'DGS3MO': 0.25, 'DGS6MO': 0.5,
        'DGS1': 1.0, 'DGS2': 2.0, 'DGS3': 3.0,
        'DGS5': 5.0, 'DGS7': 7.0, 'DGS10': 10.0,
        'DGS20': 20.0, 'DGS30': 30.0
    }
    
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.date.today()
    
    all_series = []
    
    print(f"--- Starting Data Ingestion ---")
    print(f"Targeting {len(symbol_map)} tenors from {start_date} to {end}")
    
    for symbol, tenor in symbol_map.items():
        print(f"[*] Fetching {symbol} ({tenor} year tenor)... ", end="", flush=True)
        try:
            # Fetch individual series to prevent timeout
            df_sym = web.DataReader(symbol, 'fred', start, end)
            df_sym.columns = [tenor]
            all_series.append(df_sym)
            print("Done.")
            # Small pause to be polite to FRED servers
            time.sleep(0.2) 
        except Exception as e:
            print(f"\n[!] Failed to download {symbol}: {e}")

    print("--- Processing Data ---")
    # Merge all into one matrix
    full_df = pd.concat(all_series, axis=1)
    
    # Cleaning
    full_df = full_df / 100.0  # Convert to decimals
    initial_count = len(full_df)
    full_df = full_df.dropna()
    print(f"Cleaned data: Kept {len(full_df)} of {initial_count} rows after removing NaNs.")

    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    output_path = "data/treasury_yields.csv"
    full_df.to_csv(output_path)
    print(f"[SUCCESS] Data saved to {output_path}")
    return full_df

if __name__ == "__main__":
    data = get_treasury_matrix()