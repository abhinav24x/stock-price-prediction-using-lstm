import argparse
from src.utils import fetch_stock_data, add_technical_indicators
from src.predictor import predict_future

def main():
    parser = argparse.ArgumentParser(description="Predict Future Stock Prices")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker (e.g., AAPL)")
    parser.add_argument("--days", type=int, default=1, choices=[1, 7, 30], help="Days to predict (1, 7, 30)")
    
    args = parser.parse_args()
    
    # Fetch recent data to build the last sequence
    # Need at least seq_len + window for technical indicators
    # 200 days is a safe buffer
    print(f"Fetching recent data for {args.ticker}...")
    df = fetch_stock_data(args.ticker, start_date="2023-01-01") 
    df = add_technical_indicators(df)
    
    predictions = predict_future(args.ticker, df, days_to_predict=args.days)
    
    print("\n--- Predictions ---")
    print(predictions)

if __name__ == "__main__":
    main()
