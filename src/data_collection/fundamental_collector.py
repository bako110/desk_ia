# =============================================================================
# COLLECTEUR DE DONNÉES FONDAMENTALES - NASDAQ IA TRADING                     #
# =============================================================================

import pandas as pd
import yfinance as yf
import os
from datetime import datetime
from typing import List
from .base_collector import BaseCollector

class FundamentalCollector(BaseCollector):
    """Collecteur de données fondamentales"""
    
    def __init__(self, data_path: str = "data/fundamental_data", config: dict = None):
        """
        Initialise le collecteur de données fondamentales
        
        Args:
            data_path: Chemin pour sauvegarder les données
            config: Configuration complète avec les clés API
        """
        # Initialiser le parent avec la config complète
        super().__init__(config)
        self.data_path = data_path
        self.config = config
        self.logger.info(f"FundamentalCollector initialisé avec chemin: {data_path}")
        
        # Créer la structure de dossiers
        self._create_directory_structure()
        
        # Extraire les clés API spécifiques depuis la config
        if config and 'api_keys' in config:
            api_keys = config['api_keys']
            self.alpha_vantage_key = api_keys.get('alpha_vantage', '')
            self.finnhub_key = api_keys.get('finnhub', '')
            self.yahoo_key = api_keys.get('yahoo_finance', '')  # Pas nécessaire mais pour cohérence
        else:
            self.alpha_vantage_key = ''
            self.finnhub_key = ''
            self.yahoo_key = ''
    
    def _create_directory_structure(self):
        """Crée la structure de dossiers pour les données fondamentales"""
        base_path = "data/raw/fundamental"
        directories = [
            f"{base_path}/earnings",
            f"{base_path}/financials", 
            f"{base_path}/ratios",
            f"{base_path}/estimates"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            self.logger.info(f"Dossier créé/vérifié: {directory}")
        
    def collect_data(self, symbols: List[str]) -> dict:
        """Collecte des données fondamentales et les sépare par catégorie"""
        yahoo_data = []
        alphavantage_data = []
        finnhub_data = []
        
        for symbol in symbols:
            self.logger.info(f"Collecte des fondamentaux pour {symbol}")
            
            # Yahoo Finance pour les fondamentaux de base (gratuit)
            try:
                data = self._collect_yahoo_fundamentals(symbol)
                if not data.empty:
                    yahoo_data.append(data)
            except Exception as e:
                self.logger.warning(f"Erreur fondamentaux Yahoo {symbol}: {e}")
            
            # Alpha Vantage pour données détaillées (clé disponible)
            if self.alpha_vantage_key:
                try:
                    data = self._collect_alphavantage_fundamentals(symbol)
                    if not data.empty:
                        alphavantage_data.append(data)
                except Exception as e:
                    self.logger.warning(f"Erreur fondamentaux AlphaVantage {symbol}: {e}")
            
            # Finnhub pour métriques financières (clé disponible)
            if self.finnhub_key:
                try:
                    data = self._collect_finnhub_fundamentals(symbol)
                    if not data.empty:
                        finnhub_data.append(data)
                except Exception as e:
                    self.logger.warning(f"Erreur fondamentaux Finnhub {symbol}: {e}")
            
            self.rate_limit()
        
        # Traiter et sauvegarder chaque source séparément
        result = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if yahoo_data:
            yahoo_df = pd.concat(yahoo_data, ignore_index=True)
            # Séparer par catégories
            ratios_yahoo = self._extract_ratios_data(yahoo_df)
            financials_yahoo = self._extract_financials_data(yahoo_df)
            
            # Sauvegarder dans des fichiers séparés
            self.save_to_csv(ratios_yahoo, f"data/raw/fundamental/ratios/yahoo_ratios_{timestamp}.csv")
            self.save_to_csv(financials_yahoo, f"data/raw/fundamental/financials/yahoo_financials_{timestamp}.csv")
            
            # Sauvegarde en base de données
            self.save_to_database(yahoo_df, "fundamental_data_yahoo")
            result['yahoo_data'] = yahoo_df
        
        if alphavantage_data:
            alphavantage_df = pd.concat(alphavantage_data, ignore_index=True)
            # Séparer par catégories
            ratios_av = self._extract_ratios_data(alphavantage_df)
            financials_av = self._extract_financials_data(alphavantage_df)
            
            # Sauvegarder dans des fichiers séparés
            self.save_to_csv(ratios_av, f"data/raw/fundamental/ratios/alphavantage_ratios_{timestamp}.csv")
            self.save_to_csv(financials_av, f"data/raw/fundamental/financials/alphavantage_financials_{timestamp}.csv")
            
            # Sauvegarde en base de données
            self.save_to_database(alphavantage_df, "fundamental_data_alphavantage")
            result['alphavantage_data'] = alphavantage_df
        
        if finnhub_data:
            finnhub_df = pd.concat(finnhub_data, ignore_index=True)
            # Séparer par catégories
            ratios_fh = self._extract_ratios_data(finnhub_df)
            financials_fh = self._extract_financials_data(finnhub_df)
            
            # Sauvegarder dans des fichiers séparés
            self.save_to_csv(ratios_fh, f"data/raw/fundamental/ratios/finnhub_ratios_{timestamp}.csv")
            self.save_to_csv(financials_fh, f"data/raw/fundamental/financials/finnhub_financials_{timestamp}.csv")
            
            # Sauvegarde en base de données
            self.save_to_database(finnhub_df, "fundamental_data_finnhub")
            result['finnhub_data'] = finnhub_df
        
        return result
    
    def _extract_ratios_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extrait les colonnes relatives aux ratios financiers"""
        ratio_columns = [
            'Symbol', 'Timestamp', 'Source',
            'PERatio', 'ForwardPE', 'TrailingPE', 'PEGRatio', 
            'PriceToBook', 'PriceToSales', 'ROE', 'ROA', 'ROI',
            'DebtToEquity', 'CurrentRatio', 'QuickRatio', 'CashRatio',
            'GrossMargins', 'OperatingMargins', 'ProfitMargins', 'NetMargin',
            'Beta', 'DividendYield', 'PayoutRatio', 'AssetTurnover',
            'InventoryTurnover', 'DebtEquityRatio', 'EnterpriseToRevenue',
            'EnterpriseToEbitda'
        ]
        
        # Filtrer seulement les colonnes qui existent dans le DataFrame
        available_columns = [col for col in ratio_columns if col in df.columns]
        return df[available_columns].copy()
    
    def _extract_financials_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extrait les colonnes relatives aux données financières"""
        financial_columns = [
            'Symbol', 'Timestamp', 'Source',
            'MarketCap', 'EnterpriseValue', 'TotalCash', 'TotalDebt',
            'TotalRevenue', 'RevenuePerShare', 'RevenueGrowth',
            'EarningsGrowth', 'BookValue', 'EPS', 'DilutedEPS',
            'EBITDA', 'SharesOutstanding', 'DividendPerShare',
            'QuarterlyEarningsGrowth', 'QuarterlyRevenueGrowth',
            'GrossMargin', 'OperatingMargin', 'ProfitMargin'
        ]
        
        # Filtrer seulement les colonnes qui existent dans le DataFrame
        available_columns = [col for col in financial_columns if col in df.columns]
        return df[available_columns].copy()
    
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
        """Fondamentaux via Alpha Vantage"""
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol,
            'apikey': self.alpha_vantage_key
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
        """Fondamentaux via Finnhub"""
        url = f"https://finnhub.io/api/v1/stock/metric"
        params = {
            'symbol': symbol,
            'metric': 'all',
            'token': self.finnhub_key
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
        if not self.finnhub_key:
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
                    'token': self.finnhub_key
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
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # Sauvegarder dans le dossier earnings
            self.save_to_csv(result, f"data/raw/fundamental/earnings/earnings_{timestamp}.csv")
            return result
        
        return pd.DataFrame()

    def get_yahoo_financial_statements(self, symbol: str) -> dict:
        """Récupère les états financiers via Yahoo Finance et les sauvegarde séparément"""
        try:
            ticker = yf.Ticker(symbol)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # États financiers
            financials = {
                'income_statement': ticker.financials,
                'balance_sheet': ticker.balance_sheet,
                'cash_flow': ticker.cashflow,
                'quarterly_financials': ticker.quarterly_financials,
                'quarterly_balance_sheet': ticker.quarterly_balance_sheet,
                'quarterly_cashflow': ticker.quarterly_cashflow
            }
            
            # Sauvegarder chaque état financier dans un fichier séparé
            for statement_type, data in financials.items():
                if not data.empty:
                    filename = f"data/raw/fundamental/financials/{symbol}_{statement_type}_{timestamp}.csv"
                    self.save_to_csv(data, filename)
            
            return financials
            
        except Exception as e:
            self.logger.error(f"Erreur états financiers Yahoo pour {symbol}: {e}")
            return {}

    def collect_analyst_estimates(self, symbols: List[str]) -> pd.DataFrame:
        """Collecte les estimations d'analystes et les sauvegarde"""
        all_estimates = []
        
        for symbol in symbols:
            self.logger.info(f"Collecte des estimations pour {symbol}")
            
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                if info:
                    estimates_data = {
                        'Symbol': symbol,
                        'Timestamp': datetime.now(),
                        'RecommendationMean': info.get('recommendationMean'),
                        'RecommendationKey': info.get('recommendationKey'),
                        'NumberOfAnalystOpinions': info.get('numberOfAnalystOpinions'),
                        'TargetHighPrice': info.get('targetHighPrice'),
                        'TargetLowPrice': info.get('targetLowPrice'),
                        'TargetMeanPrice': info.get('targetMeanPrice'),
                        'TargetMedianPrice': info.get('targetMedianPrice'),
                        'CurrentPrice': info.get('currentPrice'),
                        'Source': 'yahoo'
                    }
                    all_estimates.append(estimates_data)
                    
            except Exception as e:
                self.logger.error(f"Erreur estimations {symbol}: {e}")
            
            self.rate_limit()
        
        if all_estimates:
            result = pd.DataFrame(all_estimates)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # Sauvegarder dans le dossier estimates
            self.save_to_csv(result, f"data/raw/fundamental/estimates/analyst_estimates_{timestamp}.csv")
            return result
        
        return pd.DataFrame()

    def collect(self, symbols: List[str] = None):
        """Méthode principale de collecte - interface commune avec les autres collecteurs"""
        if symbols is None:
            # Symboles par défaut du NASDAQ 100
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ADBE', 'CRM']
        
        self.logger.info(f"Début de la collecte des données fondamentales pour {len(symbols)} symboles")
        
        # Log des APIs disponibles
        apis_available = []
        if self.alpha_vantage_key:
            apis_available.append("Alpha Vantage ✓")
        if self.finnhub_key:
            apis_available.append("Finnhub ✓")
        apis_available.append("Yahoo Finance (gratuit)")
        
        self.logger.info(f"APIs disponibles: {', '.join(apis_available)}")
        
        # Collecte des données fondamentales (ratios + financials séparés)
        fundamental_data = self.collect_data(symbols)
        
        # Collecte des données de résultats (dans earnings/)
        earnings_data = self.collect_earnings_data(symbols)
        
        # Collecte des estimations d'analystes (dans estimates/)
        estimates_data = self.collect_analyst_estimates(symbols)
        
        # Collecte des états financiers détaillés pour chaque symbole
        for symbol in symbols[:3]:  # Limiter à 3 pour éviter trop de fichiers
            self.get_yahoo_financial_statements(symbol)
        
        self.logger.info("Collecte des données fondamentales terminée")
        
        return {
            'fundamental_data': fundamental_data,
            'earnings_data': earnings_data,
            'estimates_data': estimates_data,
            'summary': {
                'symbols_processed': len(symbols),
                'fundamental_sources': len(fundamental_data),
                'earnings_records': len(earnings_data) if not earnings_data.empty else 0,
                'estimates_records': len(estimates_data) if not estimates_data.empty else 0
            }
        }