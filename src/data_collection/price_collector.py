# =============================================================================
# COLLECTEUR DE PRIX - NASDAQ IA TRADING (PRIORITÉ 1)
# =============================================================================

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List
from pathlib import Path
import os

# Import de la classe de base (à adapter selon votre structure)
# from .base_collector import BaseCollector, APIConfig

class PriceCollector:
    """Collecteur de données de prix avec fallback multi-sources"""
    
    def __init__(self, data_path: Path, config: dict):
        """
        Initialise le collecteur de prix
        
        Args:
            data_path: Chemin vers le dossier de données
            config: Configuration contenant les clés API
        """
        import logging
        self.logger = logging.getLogger(__name__)
        self.data_path = Path(data_path)
        self.config = config
        self.api_keys = config.get('api_keys', {})
        
        # Créer les dossiers nécessaires
        self.data_path.mkdir(parents=True, exist_ok=True)
        (self.data_path / "daily").mkdir(parents=True, exist_ok=True)
        (self.data_path / "intraday").mkdir(parents=True, exist_ok=True)
        
        # Configuration des sources de données
        self.sources = ['yahoo', 'alphavantage', 'polygon', 'finnhub']
        
        # Session pour les requêtes HTTP
        import requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        self.logger.info(f"PriceCollector initialisé avec chemin: {self.data_path}")
    
    def collect(self):
        """Méthode principale de collecte appelée par CollectorManager"""
        # Symboles par défaut à collecter
        default_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
            'META', 'NVDA', 'NFLX', 'ORCL', 'CRM'
        ]
        
        try:
            # Collecte des données journalières
            daily_data = self.collect_daily_data(default_symbols)
            if not daily_data.empty:
                self.logger.info(f"Collecté {len(daily_data)} enregistrements de prix journaliers")
            
            # Collecte des données intrajournalières
            intraday_data = self.collect_intraday_data(default_symbols[:5])  # Limité pour éviter les rate limits
            if not intraday_data.empty:
                self.logger.info(f"Collecté {len(intraday_data)} enregistrements intrajournaliers")
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la collecte des prix: {e}")
            raise
    
    def collect_daily_data(self, symbols: List[str], period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """Collecte des données de prix journalières avec fallback automatique"""
        all_data = []
        
        for symbol in symbols:
            self.logger.info(f"Collecte des prix journaliers pour {symbol}")
            
            # Essai avec Yahoo Finance (gratuit et fiable)
            try:
                data = self._collect_yahoo(symbol, period, interval)
                if not data.empty:
                    all_data.append(data)
                    self.logger.debug(f"Yahoo Finance réussi pour {symbol}")
                    continue
            except Exception as e:
                self.logger.warning(f"Yahoo Finance échoué pour {symbol}: {e}")
            
            # Fallback Alpha Vantage
            try:
                data = self._collect_alphavantage(symbol)
                if not data.empty:
                    all_data.append(data)
                    self.logger.debug(f"Alpha Vantage réussi pour {symbol}")
                    continue
            except Exception as e:
                self.logger.warning(f"Alpha Vantage échoué pour {symbol}: {e}")
            
            # Fallback Polygon
            try:
                data = self._collect_polygon(symbol)
                if not data.empty:
                    all_data.append(data)
                    self.logger.debug(f"Polygon réussi pour {symbol}")
                    continue
            except Exception as e:
                self.logger.warning(f"Polygon échoué pour {symbol}: {e}")
            
            # Fallback Finnhub
            try:
                data = self._collect_finnhub(symbol)
                if not data.empty:
                    all_data.append(data)
                    self.logger.debug(f"Finnhub réussi pour {symbol}")
                else:
                    self.logger.error(f"Toutes les sources ont échoué pour {symbol}")
            except Exception as e:
                self.logger.error(f"Tous les sources échoués pour {symbol}: {e}")
            
            # Rate limiting
            import time
            time.sleep(0.5)  # Pause entre les requêtes
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            # Sauvegarde en CSV
            filename = f"prices_daily_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = self.data_path / "daily" / filename
            self.save_to_csv(result, filepath)
            return result
        
        return pd.DataFrame()
    
    def _collect_yahoo(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Collecte via Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                return pd.DataFrame()
            
            data.reset_index(inplace=True)
            data['Symbol'] = symbol
            data['Source'] = 'yahoo'
            data['Timestamp'] = pd.to_datetime(data['Date'])
            
            return data[['Symbol', 'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Source']]
        except Exception as e:
            self.logger.error(f"Erreur Yahoo Finance pour {symbol}: {e}")
            return pd.DataFrame()
    
    def _collect_alphavantage(self, symbol: str) -> pd.DataFrame:
        """Collecte via Alpha Vantage"""
        api_key = self.api_keys.get('alpha_vantage')
        if not api_key or api_key.startswith('YOUR_'):
            self.logger.warning("Clé API Alpha Vantage non configurée")
            return pd.DataFrame()
        
        url = f"https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': symbol,
            'apikey': api_key,
            'outputsize': 'full'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code != 200:
                self.logger.error(f"Alpha Vantage API error: {response.status_code}")
                return pd.DataFrame()
                
            data = response.json()
            
            if 'Time Series (Daily)' not in data:
                if 'Error Message' in data:
                    self.logger.error(f"Alpha Vantage error: {data['Error Message']}")
                elif 'Note' in data:
                    self.logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
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
        except Exception as e:
            self.logger.error(f"Erreur Alpha Vantage pour {symbol}: {e}")
            return pd.DataFrame()
    
    def _collect_polygon(self, symbol: str) -> pd.DataFrame:
        """Collecte via Polygon"""
        api_key = self.api_keys.get('polygon')
        if not api_key or api_key.startswith('YOUR_'):
            self.logger.warning("Clé API Polygon non configurée")
            return pd.DataFrame()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # 2 ans
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        params = {'apikey': api_key}
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code != 200:
                self.logger.error(f"Polygon API error: {response.status_code}")
                return pd.DataFrame()
                
            data = response.json()
            
            if 'results' not in data:
                self.logger.warning(f"Polygon: pas de résultats pour {symbol}")
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
        except Exception as e:
            self.logger.error(f"Erreur Polygon pour {symbol}: {e}")
            return pd.DataFrame()
    
    def _collect_finnhub(self, symbol: str) -> pd.DataFrame:
        """Collecte via Finnhub"""
        api_key = self.api_keys.get('finnhub')
        if not api_key or api_key.startswith('YOUR_'):
            self.logger.warning("Clé API Finnhub non configurée")
            return pd.DataFrame()
        
        end_time = int(datetime.now().timestamp())
        start_time = int((datetime.now() - timedelta(days=730)).timestamp())
        
        url = f"https://finnhub.io/api/v1/stock/candle"
        params = {
            'symbol': symbol,
            'resolution': 'D',
            'from': start_time,
            'to': end_time,
            'token': api_key
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code != 200:
                self.logger.error(f"Finnhub API error: {response.status_code}")
                return pd.DataFrame()
                
            data = response.json()
            
            if data.get('s') != 'ok':
                self.logger.warning(f"Finnhub: pas de données pour {symbol}")
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
        except Exception as e:
            self.logger.error(f"Erreur Finnhub pour {symbol}: {e}")
            return pd.DataFrame()
    
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
            
            # Rate limiting
            import time
            time.sleep(0.5)
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            filename = f"intraday_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            filepath = self.data_path / "intraday" / filename
            self.save_to_csv(result, filepath)
            return result
        
        return pd.DataFrame()
    
    def save_to_csv(self, data: pd.DataFrame, filepath: Path):
        """Sauvegarde les données en CSV"""
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(filepath, index=False, encoding='utf-8')
            self.logger.info(f"Données sauvegardées: {filepath}")
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde CSV {filepath}: {e}")
    
    def get_latest_prices(self, symbols: List[str]) -> pd.DataFrame:
        """Récupère les derniers prix en temps réel"""
        all_data = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                latest_data = {
                    'Symbol': symbol,
                    'Timestamp': datetime.now(),
                    'Price': info.get('regularMarketPrice', 0),
                    'Change': info.get('regularMarketChange', 0),
                    'ChangePercent': info.get('regularMarketChangePercent', 0),
                    'Volume': info.get('regularMarketVolume', 0),
                    'Source': 'yahoo_realtime'
                }
                all_data.append(latest_data)
                
            except Exception as e:
                self.logger.error(f"Erreur prix temps réel {symbol}: {e}")
        
        return pd.DataFrame(all_data) if all_data else pd.DataFrame()