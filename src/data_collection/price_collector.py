# =============================================================================
# COLLECTEUR DE PRIX - NASDAQ IA TRADING (PRIORITÉ 1)
# =============================================================================

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List
from .base_collector import BaseCollector, APIConfig

class PriceCollector(BaseCollector):
    """Collecteur de données de prix avec fallback multi-sources"""
    
    def __init__(self, api_config: APIConfig):
        super().__init__(api_config)
        self.sources = ['yahoo', 'alphavantage', 'polygon', 'finnhub']
        
    def collect_data(self, symbols: List[str], period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """Collecte des données de prix avec fallback automatique"""
        all_data = []
        
        for symbol in symbols:
            self.logger.info(f"Collecte des prix pour {symbol}")
            
            # Essai avec Yahoo Finance (gratuit et fiable)
            try:
                data = self._collect_yahoo(symbol, period, interval)
                if not data.empty:
                    all_data.append(data)
                    continue
            except Exception as e:
                self.logger.warning(f"Yahoo Finance échoué pour {symbol}: {e}")
            
            # Fallback Alpha Vantage
            if not all_data or all_data[-1].empty:
                try:
                    data = self._collect_alphavantage(symbol)
                    if not data.empty:
                        all_data.append(data)
                        continue
                except Exception as e:
                    self.logger.warning(f"Alpha Vantage échoué pour {symbol}: {e}")
            
            # Fallback Polygon
            if not all_data or all_data[-1].empty:
                try:
                    data = self._collect_polygon(symbol)
                    if not data.empty:
                        all_data.append(data)
                        continue
                except Exception as e:
                    self.logger.warning(f"Polygon échoué pour {symbol}: {e}")
            
            # Fallback Finnhub
            if not all_data or all_data[-1].empty:
                try:
                    data = self._collect_finnhub(symbol)
                    if not data.empty:
                        all_data.append(data)
                except Exception as e:
                    self.logger.error(f"Tous les sources échoués pour {symbol}: {e}")
            
            self.rate_limit()
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            # Sauvegarde en base de données
            self.save_to_database(result, "price_data")
            # Sauvegarde en CSV
            self.save_to_csv(result, f"data/raw/price_data/daily/prices_{datetime.now().strftime('%Y%m%d')}.csv")
            return result
        
        return pd.DataFrame()
    
    def _collect_yahoo(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Collecte via Yahoo Finance"""
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            return pd.DataFrame()
        
        data.reset_index(inplace=True)
        data['Symbol'] = symbol
        data['Source'] = 'yahoo'
        data['Timestamp'] = pd.to_datetime(data['Date'])
        
        return data[['Symbol', 'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Source']]
    
    def _collect_alphavantage(self, symbol: str) -> pd.DataFrame:
        """Collecte via Alpha Vantage"""
        url = f"https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': symbol,
            'apikey': self.api_config.alphavantage,
            'outputsize': 'full'
        }
        
        response = self.session.get(url, params=params)
        
        if not self.handle_api_error(response, 'alphavantage'):
            return pd.DataFrame()
            
        data = response.json()
        
        if 'Time Series (Daily)' not in data:
            return pd.DataFrame()
        
        df_list = []
        for date, values in data['Time Series (Daily)'].items():
            df_list.append({
                'Symbol': symbol,
                'Timestamp': pd.to_datetime(date),
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Close': float(values['4. close']),
                'Volume': int(values['6. volume']),
                'Source': 'alphavantage'
            })
        
        return pd.DataFrame(df_list)
    
    def _collect_polygon(self, symbol: str) -> pd.DataFrame:
        """Collecte via Polygon"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # 2 ans
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        params = {'apikey': self.api_config.polygon}
        
        response = self.session.get(url, params=params)
        
        if not self.handle_api_error(response, 'polygon'):
            return pd.DataFrame()
            
        data = response.json()
        
        if 'results' not in data:
            return pd.DataFrame()
        
        df_list = []
        for result in data['results']:
            df_list.append({
                'Symbol': symbol,
                'Timestamp': pd.to_datetime(result['t'], unit='ms'),
                'Open': result['o'],
                'High': result['h'],
                'Low': result['l'],
                'Close': result['c'],
                'Volume': result['v'],
                'Source': 'polygon'
            })
        
        return pd.DataFrame(df_list)
    
    def _collect_finnhub(self, symbol: str) -> pd.DataFrame:
        """Collecte via Finnhub"""
        end_time = int(datetime.now().timestamp())
        start_time = int((datetime.now() - timedelta(days=730)).timestamp())
        
        url = f"https://finnhub.io/api/v1/stock/candle"
        params = {
            'symbol': symbol,
            'resolution': 'D',
            'from': start_time,
            'to': end_time,
            'token': self.api_config.finnhub
        }
        
        response = self.session.get(url, params=params)
        
        if not self.handle_api_error(response, 'finnhub'):
            return pd.DataFrame()
            
        data = response.json()
        
        if data.get('s') != 'ok':
            return pd.DataFrame()
        
        df_list = []
        for i in range(len(data['t'])):
            df_list.append({
                'Symbol': symbol,
                'Timestamp': pd.to_datetime(data['t'][i], unit='s'),
                'Open': data['o'][i],
                'High': data['h'][i],
                'Low': data['l'][i],
                'Close': data['c'][i],
                'Volume': data['v'][i],
                'Source': 'finnhub'
            })
        
        return pd.DataFrame(df_list)
    
    def collect_intraday_data(self, symbols: List[str], interval: str = "5m") -> pd.DataFrame:
        """Collecte des données intrajournalières"""
        all_data = []
        
        for symbol in symbols:
            self.logger.info(f"Collecte des données intrajournalières pour {symbol}")
            
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d", interval=interval)
                
                if not data.empty:
                    data.reset_index(inplace=True)
                    data['Symbol'] = symbol
                    data['Source'] = 'yahoo'
                    data['Timestamp'] = pd.to_datetime(data['Datetime'])
                    
                    intraday_data = data[['Symbol', 'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Source']]
                    all_data.append(intraday_data)
                    
            except Exception as e:
                self.logger.error(f"Erreur données intrajournalières {symbol}: {e}")
            
            self.rate_limit()
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            self.save_to_csv(result, f"data/raw/price_data/intraday/intraday_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
            return result
        
        return pd.DataFrame()