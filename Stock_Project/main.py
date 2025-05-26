import yfinance as yf
import pandas as pd

tickers = ["AAPL", "NVDA", "MSFT", "TSLA", "WGS", "TSSI", "CVNA"]
start_date = "2023-04-15"
end_date = "2024-04-15"

results = []

for ticker in tickers:
    print(f"Downloading: {ticker}")
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

        if df.empty or 'Close' not in df.columns:
            print(f"No data for {ticker}")
            continue

        close_prices = df['Close'].dropna()
        if close_prices.empty:
            print(f"No valid close prices for {ticker}")
            continue

        start_price = close_prices.iloc[0]
        end_price = close_prices.iloc[-1]
        pct_return = ((end_price / start_price) - 1) * 100

        if pct_return >= 900:
            results.append({
                "Ticker": ticker,
                "Start Price": round(start_price, 2),
                "End Price": round(end_price, 2),
                "1Y Return (%)": round(pct_return, 2)
            })

    except Exception as e:
        print(f"Error with {ticker}: {e}")

# Final output
df_results = pd.DataFrame(results)
print(df_results)
