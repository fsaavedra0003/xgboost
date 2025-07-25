# src/fetch_data.py
import yfinance as yf
import pandas as pd
import argparse

def download_stock_data(ticker, start, end, output_path):
    data = yf.download(ticker, start=start, end=end)
    data.to_csv(output_path)
    print(f"Saved data to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--output", type=str, default="data/stock_data.csv")

    args = parser.parse_args()
    download_stock_data(args.ticker, args.start, args.end, args.output)
