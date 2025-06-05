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
    
    def __init__(self, data_path: str = "data/fundamental_data", config: dict = None):
        """
        Initialise le collecteur de données fondamentales
        
        Args:
            data_path: Chemin pour sauvegarder les données
            config: Configuration complète avec les clés API
        """
        # Créer l'objet APIConfig à partir de la configuration
        if config:
            api_config = APIConfig(
                alpha_vantage=config.get('fundamental_apis', {}).get('alpha_vantage', ''),
                finnhub=config.get('fundamental_apis', {}).get('finnhub', ''),
                yahoo_finance="",  # Yahoo Finance ne nécessite pas de clé
                bloomberg="",      # Pas de clé disponible - désactivé
                refinitiv=""       # Pas de clé disponible - désactivé
            )
        else:
            api_config = APIConfig()
            
        super().__init__(api_config)
        self.data_path = data_path
        self.config = config
        self.logger.info(f"FundamentalCollector initialisé avec chemin: {data_path}")
        
    def collect_data(self, symbols: List[str]) -> pd.DataFrame:
        """Collecte des données fondamentales"""
        all_data = []
        
        for symbol in symbols:
            self.logger.info(f"Collecte des fondamentaux pour {symbol}")
            
            # Yahoo Finance pour les fondamentaux de base (gratuit)
            try:
                data = self._collect_yahoo_fundamentals(symbol)
                if not data.empty:
                    all_data.append(data)
            except Exception as e:
                self.logger.warning(f"Erreur fondamentaux Yahoo {symbol}: {e}")
            
            # Alpha Vantage pour données détaillées (clé disponible)
            if self.api_config.alpha_vantage:
                try:
                    data = self._collect_alphavantage_fundamentals(symbol)
                    if not data.empty:
                        all_data.append(data)
                except Exception as e:
                    self.logger.warning(f"Erreur fondamentaux AlphaVantage {symbol}: {e}")
            
            # Finnhub pour métriques financières (clé disponible)
            if self.api_config.finnhub:
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
        """Fondamentaux via Yahoo Finance (gratuit, pas de clé requise)"""
        try:
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
                'BookValue': info.get('bookValue'),
                'EnterpriseValue': info.get('enterpriseValue'),
                'EnterpriseToRevenue': info.get('enterpriseToRevenue'),
                'EnterpriseToEbitda': info.get('enterpriseToEbitda'),
                'TotalCash': info.get('totalCash'),
                'TotalDebt': info.get('totalDebt'),
                'TotalRevenue': info.get('totalRevenue'),
                'RevenuePerShare': info.get('revenuePerShare'),
                'Source': 'yahoo'
            }
            
            return pd.DataFrame([fundamental_data])
            
        except Exception as e:
            self.logger.error(f"Erreur Yahoo Finance pour {symbol}: {e}")
            return pd.DataFrame()
    
    def _collect_alphavantage_fundamentals(self, symbol: str) -> pd.DataFrame:
        """Fondamentaux via Alpha Vantage (clé: RU6W0PWAUZ0JYD0A)"""
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol,
            'apikey': self.api_config.alpha_vantage
        }
        
        try:
            response = self.session.get(url, params=params)
            
            if not self.handle_api_error(response, 'alphavantage'):
                return pd.DataFrame()
                
            data = response.json()
            
            # Vérifier les erreurs API Alpha Vantage
            if 'Error Message' in data or 'Note' in data:
                self.logger.warning(f"Erreur Alpha Vantage pour {symbol}: {data}")
                return pd.DataFrame()
            
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
                'Beta': self._safe_float(data.get('Beta')),
                'DividendPerShare': self._safe_float(data.get('DividendPerShare')),
                'DividendYield': self._safe_float(data.get('DividendYield')),
                'EPS': self._safe_float(data.get('EPS')),
                'BookValue': self._safe_float(data.get('BookValue')),
                'SharesOutstanding': self._safe_float(data.get('SharesOutstanding')),
                '52WeekHigh': self._safe_float(data.get('52WeekHigh')),
                '52WeekLow': self._safe_float(data.get('52WeekLow')),
                'Source': 'alphavantage'
            }
            
            return pd.DataFrame([fundamental_data])
            
        except Exception as e:
            self.logger.error(f"Erreur Alpha Vantage pour {symbol}: {e}")
            return pd.DataFrame()
    
    def _collect_finnhub_fundamentals(self, symbol: str) -> pd.DataFrame:
        """Fondamentaux via Finnhub (clé: d0ng2fpr01qi1cve64bgd0ng2fpr01qi1cve64c0)"""
        url = f"https://finnhub.io/api/v1/stock/metric"
        params = {
            'symbol': symbol,
            'metric': 'all',
            'token': self.api_config.finnhub
        }
        
        try:
            response = self.session.get(url, params=params)
            
            if not self.handle_api_error(response, 'finnhub'):
                return pd.DataFrame()
                
            data = response.json()
            
            # Vérifier les erreurs API Finnhub
            if 'error' in data:
                self.logger.warning(f"Erreur Finnhub pour {symbol}: {data['error']}")
                return pd.DataFrame()
            
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
                'DebtEquityRatio': metrics.get('totalDebt/totalEquityQuarterly'),
                'CashRatio': metrics.get('cashRatioQuarterly'),
                'DividendYieldIndicatedAnnual': metrics.get('dividendYieldIndicatedAnnual'),
                'Source': 'finnhub'
            }
            
            return pd.DataFrame([fundamental_data])
            
        except Exception as e:
            self.logger.error(f"Erreur Finnhub pour {symbol}: {e}")
            return pd.DataFrame()
    
    def collect_earnings_data(self, symbols: List[str]) -> pd.DataFrame:
        """Collecte des données de résultats trimestriels via Finnhub"""
        if not self.api_config.finnhub:
            self.logger.warning("Clé Finnhub non disponible pour la collecte des résultats")
            return pd.DataFrame()
            
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
                    
                    if isinstance(data, list):
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
                                'Timestamp': datetime.now(),
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

    def get_yahoo_financial_statements(self, symbol: str) -> dict:
        """Récupère les états financiers via Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            
            # États financiers
            financials = {
                'income_statement': ticker.financials,
                'balance_sheet': ticker.balance_sheet,
                'cash_flow': ticker.cashflow,
                'quarterly_financials': ticker.quarterly_financials,
                'quarterly_balance_sheet': ticker.quarterly_balance_sheet,
                'quarterly_cashflow': ticker.quarterly_cashflow
            }
            
            return financials
            
        except Exception as e:
            self.logger.error(f"Erreur états financiers Yahoo pour {symbol}: {e}")
            return {}

    def collect(self, symbols: List[str] = None):
        """Méthode principale de collecte - interface commune avec les autres collecteurs"""
        if symbols is None:
            # Symboles par défaut du NASDAQ 100
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ADBE', 'CRM']
        
        self.logger.info(f"Début de la collecte des données fondamentales pour {len(symbols)} symboles")
        self.logger.info(f"APIs disponibles: Yahoo Finance (gratuit), Alpha Vantage ✓, Finnhub ✓")
        
        # Collecte des données fondamentales
        fundamental_data = self.collect_data(symbols)
        
        # Collecte des données de résultats
        earnings_data = self.collect_earnings_data(symbols)
        
        self.logger.info("Collecte des données fondamentales terminée")
        
        return {
            'fundamental_data': fundamental_data,
            'earnings_data': earnings_data,
            'summary': {
                'symbols_processed': len(symbols),
                'fundamental_records': len(fundamental_data) if not fundamental_data.empty else 0,
                'earnings_records': len(earnings_data) if not earnings_data.empty else 0
            }
        }