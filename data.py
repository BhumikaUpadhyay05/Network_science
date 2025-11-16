import pandas as pd
import numpy as np
import yfinance as yf
import os
from tqdm import tqdm

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
TICKER_FILE = "ind_nifty100list.csv"      # Input file
RAW_DATA_FOLDER = "price_data"            # Folder to save per-stock files
OUTPUT_CLOSE = "nifty100_combined_1year.csv" 
OUTPUT_RETURNS = "nifty100_logreturns.csv"

START = "2023-10-01"   # 1 year period
END   = "2024-10-01"
# ----------------------------------------------------------------------

# Create folder if needed
os.makedirs(RAW_DATA_FOLDER, exist_ok=True)

# Load tickers
df = pd.read_csv(TICKER_FILE)
# Adjust column name if needed
if "Symbol" in df.columns:
    tickers = df["Symbol"].tolist()
else:
    tickers = df.iloc[:,0].tolist()

print(f"Loaded {len(tickers)} tickers")

# Yahoo-specific tickers
tickers_ns = [t + ".NS" for t in tickers]

# Combined DataFrame
combined = pd.DataFrame()

# ----------------------------------------------------------------------
# DOWNLOAD OR LOAD CACHED PRICE DATA
# ----------------------------------------------------------------------
for t_raw, t in tqdm(zip(tickers, tickers_ns), total=len(tickers)):
    
    save_path = os.path.join(RAW_DATA_FOLDER, f"{t_raw}.csv")
    
    # Load from local file if exists
    if os.path.exists(save_path):
        df_price = pd.read_csv(save_path, index_col=0, parse_dates=True)
    else:
        try:
            yf_ticker = yf.Ticker(t)
            df_price = yf_ticker.history(start=START, end=END, interval="1d")
            df_price.to_csv(save_path)
        except Exception as e:
            print(f"Error downloading {t}: {e}")
            continue
    
    if "Close" not in df_price.columns:
        continue

    combined[t_raw] = df_price["Close"]

# ----------------------------------------------------------------------
# CLEANING DATA
# ----------------------------------------------------------------------

# Drop stocks with >20% missing values
missing_fraction = combined.isna().mean()
keep_stocks = missing_fraction[missing_fraction <= 0.2].index.tolist()
combined = combined[keep_stocks]

# Fill missing
combined = combined.fillna(method="ffill").fillna(method="bfill")
combined.to_csv(OUTPUT_CLOSE)

print(f"\nSaved cleaned close-price matrix as: {OUTPUT_CLOSE}")

# ----------------------------------------------------------------------
# LOG RETURNS
# ----------------------------------------------------------------------

logreturns = np.log(combined / combined.shift(1)).dropna() #log(P_t / P_{t-1})
logreturns.to_csv(OUTPUT_RETURNS)

print(f"Saved log-return matrix as: {OUTPUT_RETURNS}")
