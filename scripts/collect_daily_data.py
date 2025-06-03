from src.data_collection.price_collector import PriceCollector
from datetime import datetime

def main():
    collector = PriceCollector()
    today = datetime.today().strftime('%Y-%m-%d')
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    for symbol in symbols:
        collector.fetch_daily_prices(symbol, start_date="2023-01-01", end_date=today)

if __name__ == "__main__":
    main()
