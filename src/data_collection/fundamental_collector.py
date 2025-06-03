# =============================================================================
# COLLECTEUR DE DONNÉES FONDAMENTALES - NASDAQ IA TRADING                     #
# =============================================================================

import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import List
from .base_collector import BaseCollector, APIConfig

class FundamentalCollector(BaseCollector):
    """Collecteur de données fondamentales"""
    
    def __init__(self, api_config: APIConfig, data_path: str = "data/fundamental_data"):
        """
        Initialise le collecteur de données fondamentales
        
        Args:
            api_config: Configuration des clés API
            data_path: Chemin pour sauvegarder les données
        """
        super().__init__(api_config)
        self.data_path = data_path
        self.logger.info(f"FundamentalCollector initialisé avec chemin: {data_path}")
        
    def collect_data(self, symbols: List[str]) -> pd.DataFrame:
        """Collecte des données fondamentales"""
        all_data = []
        
        for symbol in symbols:
            self.logger.info(f"Collecte des fondamentaux pour {symbol}")
            
            # Yahoo Finance pour les fondamentaux de base
            try:
                data = self._collect_yahoo_fundamentals(symbol)
                if not data.empty:
                    all_data.append(data)
            except Exception as e:
                self.logger.warning(f"Erreur fondamentaux Yahoo {symbol}: {e}")
            
            # Alpha Vantage pour données détaillées
            try:
                data = self._collect_alphavantage_fundamentals(symbol)
                if not data.empty:
                    all_data.append(data)
            except Exception as e:
                self.logger.warning(f"Erreur fondamentaux AlphaVantage {symbol}: {e}")
            
            # Finnhub pour métriques financières
            try:
                data = self._collect_finnhub_fundamentals(symbol)
                if not data.empty:
                    all_data.append(data)
            except Exception as e:
                self.logger.warning(f"Erreur fondamentaux Finnhub {symbol}: {e}")
            
            self.rate_limit()
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            # Sauvegarde en base de données
            self.save_to_database(result, "fundamental_data")
            # Sauvegarde en CSV
            self.save_to_csv(result, f"data/raw/fundamental/fundamentals_{datetime.now().strftime('%Y%m%d')}.csv")
            return result
        
        return pd.DataFrame()
    
    def _collect_yahoo_fundamentals(self, symbol: str) -> pd.DataFrame:
        """Fondamentaux via Yahoo Finance"""
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        if not info:
            return pd.DataFrame()
        
        fundamental_data = {
            'Symbol': symbol,
            'Timestamp': datetime.now(),
            'MarketCap': info.get('marketCap'),
            'PERatio': info.get('trailingPE'),
            'ForwardPE': info.get('forwardPE'),
            'PEGRatio': info.get('pegRatio'),
            'PriceToBook': info.get('priceToBook'),
            'PriceToSales': info.get('priceToSalesTrailing12Months'),
            'ROE': info.get('returnOnEquity'),
            'ROA': info.get('returnOnAssets'),
            'ROI': info.get('returnOnInvestment'),
            'DebtToEquity': info.get('debtToEquity'),
            'CurrentRatio': info.get('currentRatio'),
            'QuickRatio': info.get('quickRatio'),
            'GrossMargins': info.get('grossMargins'),
            'OperatingMargins': info.get('operatingMargins'),
            'ProfitMargins': info.get('profitMargins'),
            'RevenueGrowth': info.get('revenueGrowth'),
            'EarningsGrowth': info.get('earningsGrowth'),
            'Beta': info.get('beta'),
            'DividendYield': info.get('dividendYield'),
            'PayoutRatio': info.get('payoutRatio'),
            'RecommendationMean': info.get('recommendationMean'),
            'TargetHighPrice': info.get('targetHighPrice'),
            'TargetLowPrice': info.get('targetLowPrice'),
            'TargetMeanPrice': info.get('targetMeanPrice'),
            'Source': 'yahoo'
        }
        
        return pd.DataFrame([fundamental_data])
    
    def _collect_alphavantage_fundamentals(self, symbol: str) -> pd.DataFrame:
        """Fondamentaux via Alpha Vantage"""
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol,
            'apikey': self.api_config.alphavantage
        }
        
        response = self.session.get(url, params=params)
        
        if not self.handle_api_error(response, 'alphavantage'):
            return pd.DataFrame()
            
        data = response.json()
        
        if not data or 'Symbol' not in data:
            return pd.DataFrame()
        
        fundamental_data = {
            'Symbol': symbol,
            'Timestamp': datetime.now(),
            'MarketCap': self._safe_float(data.get('MarketCapitalization')),
            'PERatio': self._safe_float(data.get('PERatio')),
            'PEGRatio': self._safe_float(data.get('PEGRatio')),
            'PriceToBook': self._safe_float(data.get('PriceToBookRatio')),
            'PriceToSales': self._safe_float(data.get('PriceToSalesRatioTTM')),
            'ROE': self._safe_float(data.get('ReturnOnEquityTTM')),
            'ROA': self._safe_float(data.get('ReturnOnAssetsTTM')),
            'RevenuePerShare': self._safe_float(data.get('RevenuePerShareTTM')),
            'ProfitMargin': self._safe_float(data.get('ProfitMargin')),
            'OperatingMargin': self._safe_float(data.get('OperatingMarginTTM')),
            'GrossMargin': self._safe_float(data.get('GrossProfitTTM')),
            'EBITDA': self._safe_float(data.get('EBITDA')),
            'DilutedEPS': self._safe_float(data.get('DilutedEPSTTM')),
            'QuarterlyEarningsGrowth': self._safe_float(data.get('QuarterlyEarningsGrowthYOY')),
            'QuarterlyRevenueGrowth': self._safe_float(data.get('QuarterlyRevenueGrowthYOY')),
            'AnalystTargetPrice': self._safe_float(data.get('AnalystTargetPrice')),
            'TrailingPE': self._safe_float(data.get('TrailingPE')),
            'ForwardPE': self._safe_float(data.get('ForwardPE')),
            'PriceToSalesRatio': self._safe_float(data.get('PriceToSalesRatioTTM')),
            'Beta': self._safe_float(data.get('Beta')),
            'DividendPerShare': self._safe_float(data.get('DividendPerShare')),
            'DividendYield': self._safe_float(data.get('DividendYield')),
            'Source': 'alphavantage'
        }
        
        return pd.DataFrame([fundamental_data])
    
    def _collect_finnhub_fundamentals(self, symbol: str) -> pd.DataFrame:
        """Fondamentaux via Finnhub"""
        url = f"https://finnhub.io/api/v1/stock/metric"
        params = {
            'symbol': symbol,
            'metric': 'all',
            'token': self.api_config.finnhub
        }
        
        response = self.session.get(url, params=params)
        
        if not self.handle_api_error(response, 'finnhub'):
            return pd.DataFrame()
            
        data = response.json()
        
        if 'metric' not in data:
            return pd.DataFrame()
        
        metrics = data['metric']
        fundamental_data = {
            'Symbol': symbol,
            'Timestamp': datetime.now(),
            'PERatio': metrics.get('peBasicExclExtraTTM'),
            'PriceToBook': metrics.get('pbQuarterly'),
            'PriceToSales': metrics.get('psQuarterly'),
            'ROE': metrics.get('roeRfy'),
            'ROA': metrics.get('roaRfy'),
            'ROI': metrics.get('roiRfy'),
            'CurrentRatio': metrics.get('currentRatioQuarterly'),
            'QuickRatio': metrics.get('quickRatioQuarterly'),
            'GrossMargin': metrics.get('grossMarginTTM'),
            'OperatingMargin': metrics.get('operatingMarginTTM'),
            'NetMargin': metrics.get('netMarginTTM'),
            'AssetTurnover': metrics.get('assetTurnoverTTM'),
            'InventoryTurnover': metrics.get('inventoryTurnoverTTM'),
            'RevenueGrowth': metrics.get('revenueGrowthTTM'),
            'EarningsGrowth': metrics.get('epsGrowthTTM'),
            'Beta': metrics.get('beta'),
            'Source': 'finnhub'
        }
        
        return pd.DataFrame([fundamental_data])
    
    def collect_earnings_data(self, symbols: List[str]) -> pd.DataFrame:
        """Collecte des données de résultats trimestriels"""
        all_earnings = []
        
        for symbol in symbols:
            self.logger.info(f"Collecte des résultats pour {symbol}")
            
            try:
                # Résultats via Finnhub
                url = f"https://finnhub.io/api/v1/stock/earnings"
                params = {
                    'symbol': symbol,
                    'token': self.api_config.finnhub
                }
                
                response = self.session.get(url, params=params)
                
                if self.handle_api_error(response, 'finnhub'):
                    data = response.json()
                    
                    for earning in data:
                        earnings_data = {
                            'Symbol': symbol,
                            'Period': earning.get('period'),
                            'Year': earning.get('year'),
                            'Quarter': earning.get('quarter'),
                            'Actual': earning.get('actual'),
                            'Estimate': earning.get('estimate'),
                            'Surprise': earning.get('surprise'),
                            'SurprisePercent': earning.get('surprisePercent'),
                            'Source': 'finnhub'
                        }
                        all_earnings.append(earnings_data)
                        
            except Exception as e:
                self.logger.error(f"Erreur résultats {symbol}: {e}")
            
            self.rate_limit()
        
        if all_earnings:
            result = pd.DataFrame(all_earnings)
            self.save_to_csv(result, f"data/raw/fundamental/earnings/earnings_{datetime.now().strftime('%Y%m%d')}.csv")
            return result
        
        return pd.DataFrame()