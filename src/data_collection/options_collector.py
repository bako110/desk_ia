#!/usr/bin/env python3
"""
NASDAQ IA Trading - Options Collector
Collecte les données d'options: flux, volatilité, gamma exposure, etc.
"""

import os
import time
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import requests
import yfinance as yf
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import scipy.stats as stats
from math import log, sqrt, exp
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/options_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('OptionsCollector')

class OptionsDataCollector:
    """Collecteur spécialisé pour les données d'options"""
    
    def __init__(self):
        self.data_path = Path(os.environ.get('DATA_PATH', '/app/data/options'))
        self.symbols = os.environ.get('SYMBOLS', 'AAPL,GOOGL,MSFT,TSLA,NVDA,QQQ,SPY').split(',')
        
        # Configuration des APIs
        self.api_keys = {
            'alphavantage': os.environ.get('ALPHAVANTAGE_API_KEY', 'RU6W0PWAUZ0JYD0A'),
            'polygon': os.environ.get('POLYGON_API_KEY', '6_TgTH0XU8AdgToOqMqfrEsE4PkwRJda'),
            'finnhub': os.environ.get('FINNHUB_API_KEY', 'd0ng2fpr01qi1cve64bgd0ng2fpr01qi1cve64c0'),
            'iexcloud': os.environ.get('IEXCLOUD_API_KEY', 'TONE_KEY_IEXCLOUD_ICI')
        }
        
        self.rate_limits = {
            'yahoo_finance': int(os.environ.get('YAHOO_RATE_LIMIT', 1000)),
            'polygon': int(os.environ.get('POLYGON_RATE_LIMIT', 100)),
            'alphavantage': int(os.environ.get('ALPHAVANTAGE_RATE_LIMIT', 5))
        }
        
        # Configuration des paramètres d'options
        self.risk_free_rate = 0.05  # Taux sans risque
        self.trading_days_per_year = 252
        
        self._setup_directories()
    
    def _setup_directories(self):
        """Crée la structure des répertoires pour les données d'options"""
        directories = [
            'options_chains',
            'unusual_options_activity',
            'implied_volatility',
            'options_flow',
            'gamma_exposure',
            'delta_hedging',
            'volatility_smile',
            'options_volume',
            'put_call_ratio',
            'max_pain',
            'dark_pool_prints',
            'block_trades',
            'options_spreads',
            'volatility_surface',
            'greeks_analysis'
        ]
        
        for directory in directories:
            (self.data_path / directory).mkdir(parents=True, exist_ok=True)
            
        logger.info(f"📁 Structure options créée dans {self.data_path}")
    
    def collect_options_chains(self):
        """Collecte les chaînes d'options complètes"""
        logger.info("⛓️ Collecte des chaînes d'options...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                
                # Récupération des dates d'expiration
                expirations = ticker.options
                
                if not expirations:
                    logger.warning(f"⚠️ Pas d'options disponibles pour {symbol}")
                    continue
                
                # Collecte pour les 3 prochaines expirations
                symbol_options_data = {
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'current_price': self._get_current_price(symbol),
                    'expirations': []
                }
                
                for exp_date in expirations[:3]:  # Limite aux 3 prochaines
                    try:
                        options_chain = ticker.option_chain(exp_date)
                        
                        calls_data = options_chain.calls.to_dict('records') if not options_chain.calls.empty else []
                        puts_data = options_chain.puts.to_dict('records') if not options_chain.puts.empty else []
                        
                        # Calcul des Greeks
                        calls_with_greeks = self._calculate_greeks(calls_data, symbol_options_data['current_price'], exp_date, 'call')
                        puts_with_greeks = self._calculate_greeks(puts_data, symbol_options_data['current_price'], exp_date, 'put')
                        
                        exp_data = {
                            'expiration_date': exp_date,
                            'days_to_expiry': (datetime.strptime(exp_date, '%Y-%m-%d').date() - date.today()).days,
                            'calls': calls_with_greeks,
                            'puts': puts_with_greeks,
                            'total_call_volume': sum(c.get('volume', 0) or 0 for c in calls_data),
                            'total_put_volume': sum(p.get('volume', 0) or 0 for p in puts_data),
                            'put_call_ratio': self._calculate_put_call_ratio(calls_data, puts_data),
                            'max_pain': self._calculate_max_pain(calls_data, puts_data, symbol_options_data['current_price'])
                        }
                        
                        symbol_options_data['expirations'].append(exp_data)
                        
                    except Exception as e:
                        logger.error(f"❌ Erreur collecte expiration {exp_date} pour {symbol}: {e}")
                
                # Sauvegarde
                file_path = self.data_path / f'options_chains/{symbol}_chain_{timestamp}.json'
                with open(file_path, 'w') as f:
                    json.dump(symbol_options_data, f, indent=2, default=str)
                
                logger.info(f"⛓️ {symbol}: {len(symbol_options_data['expirations'])} expirations collectées")
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"❌ Erreur collecte chaîne options {symbol}: {e}")
    
    def _get_current_price(self, symbol):
        """Récupère le prix actuel du sous-jacent"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1d')
            return float(hist['Close'].iloc[-1]) if not hist.empty else 0
        except:
            return 0
    
    def _calculate_greeks(self, options_data, spot_price, expiration_date, option_type):
        """Calcule les Greeks pour les options"""
        if not options_data or spot_price == 0:
            return options_data
        
        exp_date = datetime.strptime(expiration_date, '%Y-%m-%d').date()
        time_to_expiry = (exp_date - date.today()).days / 365.0
        
        if time_to_expiry <= 0:
            return options_data
        
        for option in options_data:
            try:
                strike = option.get('strike', 0)
                implied_vol = option.get('impliedVolatility', 0)
                
                if strike > 0 and implied_vol > 0:
                    greeks = self._black_scholes_greeks(
                        spot_price, strike, time_to_expiry, 
                        self.risk_free_rate, implied_vol, option_type
                    )
                    option.update(greeks)
                    
            except Exception as e:
                logger.warning(f"⚠️ Erreur calcul Greeks: {e}")
        
        return options_data
    
    def _black_scholes_greeks(self, S, K, T, r, sigma, option_type):
        """Calcule les Greeks avec le modèle Black-Scholes"""
        try:
            d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
            d2 = d1 - sigma*sqrt(T)
            
            # Delta
            if option_type == 'call':
                delta = stats.norm.cdf(d1)
            else:
                delta = -stats.norm.cdf(-d1)
            
            # Gamma
            gamma = stats.norm.pdf(d1) / (S * sigma * sqrt(T))
            
            # Theta
            if option_type == 'call':
                theta = (-S * stats.norm.pdf(d1) * sigma / (2 * sqrt(T)) 
                        - r * K * exp(-r*T) * stats.norm.cdf(d2)) / 365
            else:
                theta = (-S * stats.norm.pdf(d1) * sigma / (2 * sqrt(T)) 
                        + r * K * exp(-r*T) * stats.norm.cdf(-d2)) / 365
            
            # Vega
            vega = S * stats.norm.pdf(d1) * sqrt(T) / 100
            
            # Rho
            if option_type == 'call':
                rho = K * T * exp(-r * T) * stats.norm.cdf(d2) / 100
            else:
                rho = -K * T * exp(-r * T) * stats.norm.cdf(-d2) / 100

            return {
                'delta': round(delta, 4),
                'gamma': round(gamma, 4),
                'theta': round(theta, 4),
                'vega': round(vega, 4),
                'rho': round(rho, 4)
            }
        except Exception as e:
            logger.error(f"Erreur dans le calcul Black-Scholes Greeks: {e}")
            return {}

    def _calculate_put_call_ratio(self, calls, puts):
        """Calcule le Put/Call Ratio"""
        total_calls = sum(c.get('volume', 0) or 0 for c in calls)
        total_puts = sum(p.get('volume', 0) or 0 for p in puts)
        if total_calls == 0:
            return float('inf') if total_puts > 0 else 0
        return round(total_puts / total_calls, 4)

    def _calculate_max_pain(self, calls, puts, spot_price):
        """Calcule le Max Pain (prix où la perte totale des options est minimisée)"""
        try:
            all_strikes = sorted(set([c['strike'] for c in calls] + [p['strike'] for p in puts]))
            min_pain = float('inf')
            max_pain_strike = 0

            for strike in all_strikes:
                call_pain = sum([(max(0, c['strike'] - strike) * c.get('openInterest', 0)) for c in calls])
                put_pain = sum([(max(0, strike - p['strike']) * p.get('openInterest', 0)) for p in puts])
                total_pain = call_pain + put_pain
                if total_pain < min_pain:
                    min_pain = total_pain
                    max_pain_strike = strike
            return max_pain_strike
        except Exception as e:
            logger.warning(f"⚠️ Erreur calcul Max Pain: {e}")
            return None
