"""
NASDAQ IA Trading - Volume Collector
Collecte les données de volume de trading en temps réel
Version corrigée et optimisée
"""

import os
import time
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import yfinance as yf
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
from alpha_vantage.timeseries import TimeSeries
import finnhub
import yaml
from typing import Dict, List, Optional, Any

# Configuration du logging
def setup_logging():
    """Configure le système de logging"""
    logger = logging.getLogger('VolumeCollector')
    logger.setLevel(logging.INFO)
    
    # Évite les doublons de handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Création des handlers
    log_dir = Path('/app/logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_dir / 'volume_collector.log')
    stream_handler = logging.StreamHandler()
    
    # Création du formatter et ajout aux handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    # Ajout des handlers au logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

logger = setup_logging()

class VolumeCollector:
    """Collecteur spécialisé pour les données de volume de trading"""

    @staticmethod
    def load_api_keys(yaml_path: str = "config/api_key.yaml") -> Dict[str, str]:
        """Charge les clés API depuis le fichier YAML"""
        try:
            config_path = Path(yaml_path)
            if not config_path.exists():
                logger.warning(f"⚠️ Fichier de configuration {yaml_path} introuvable")
                return {}
                
            with open(config_path, 'r') as f:
                keys = yaml.safe_load(f)
                logger.info("🔑 Clés API chargées avec succès")
                return keys or {}
        except Exception as e:
            logger.error(f"❌ Erreur chargement des clés API: {e}")
            return {}

    def __init__(self):
        """Initialise le collecteur de volume"""
        self.data_path = Path(os.environ.get('DATA_PATH', '/app/data/volumes'))
        self.symbols = [s.strip().upper() for s in os.environ.get('SYMBOLS', 'AAPL,GOOGL,MSFT,TSLA,NVDA,QQQ,SPY').split(',')]

        # Configuration des APIs
        self.api_keys = self.load_api_keys()

        # Rate limits avec valeurs par défaut sécurisées
        self.rate_limits = {
            'alphavantage': int(os.environ.get('ALPHAVANTAGE_RATE_LIMIT', 5)),
            'polygon': int(os.environ.get('POLYGON_RATE_LIMIT', 100)),
            'finnhub': int(os.environ.get('FINNHUB_RATE_LIMIT', 60)),
            'yahoo_finance': int(os.environ.get('YAHOO_RATE_LIMIT', 200))  # Plus conservateur
        }

        self._setup_directories()
        self._initialize_clients()

    def _setup_directories(self):
        """Crée la structure des répertoires pour les données de volume"""
        directories = [
            'intraday_volume',
            'daily_volume',
            'unusual_volume',
            'volume_profile',
            'volume_analysis',
            'institutional_volume',
            'dark_pool_volume',
            'block_trades',
            'volume_weighted_prices'
        ]

        for directory in directories:
            dir_path = self.data_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"📁 Structure de répertoires créée dans {self.data_path}")

    def _initialize_clients(self):
        """Initialise les clients API"""
        self.av_client = None
        self.finnhub_client = None
        
        try:
            if self.api_keys.get('alphavantage'):
                self.av_client = TimeSeries(key=self.api_keys['alphavantage'])
                logger.info("✅ Client Alpha Vantage initialisé")
                
            if self.api_keys.get('finnhub'):
                self.finnhub_client = finnhub.Client(api_key=self.api_keys['finnhub'])
                logger.info("✅ Client Finnhub initialisé")

            logger.info("🔌 Clients API initialisés avec succès")
        except Exception as e:
            logger.error(f"❌ Erreur initialisation clients API: {e}")

    def collect_intraday_volume(self) -> List[Dict[str, Any]]:
        """Collecte les volumes intraday pour tous les symboles"""
        logger.info("📊 Début collecte volumes intraday...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results = []

        with ThreadPoolExecutor(max_workers=3) as executor:  # Réduit pour éviter rate limiting
            futures = {
                executor.submit(self._collect_symbol_intraday_volume, symbol, timestamp): symbol
                for symbol in self.symbols
            }
            
            for future in futures:
                symbol = futures[future]
                try:
                    result = future.result(timeout=30)  # Timeout de 30 secondes
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"❌ Erreur collecte {symbol}: {e}")

        # Sauvegarde consolidée
        if results:
            consolidated_file = self.data_path / f'intraday_volume/consolidated_volume_{timestamp}.json'
            with open(consolidated_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

        logger.info(f"✅ Collecte intraday terminée - {len(results)} symboles traités")
        return results

    def _collect_symbol_intraday_volume(self, symbol: str, timestamp: str) -> Optional[Dict[str, Any]]:
        """Collecte les données de volume pour un symbole spécifique"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1d', interval='1m')
            
            if hist.empty:
                logger.warning(f"⚠️ Aucune donnée Yahoo Finance pour {symbol}")
                return None

            # Nettoyage des données
            hist = hist.dropna()
            if len(hist) == 0:
                return None

            # Calcul des métriques de volume
            current_volume = int(hist['Volume'].iloc[-1]) if len(hist) > 0 else 0
            total_volume = int(hist['Volume'].sum())
            avg_volume = float(hist['Volume'].mean())
            max_volume = int(hist['Volume'].max())
            
            # VWAP (Volume Weighted Average Price)
            vwap = float((hist['Close'] * hist['Volume']).sum() / hist['Volume'].sum()) if hist['Volume'].sum() > 0 else 0

            volume_data = {
                'symbol': symbol,
                'timestamp': timestamp,
                'source': 'yahoo_finance',
                'interval': '1m',
                'data': {
                    'current_volume': current_volume,
                    'total_volume_today': total_volume,
                    'avg_volume_per_minute': avg_volume,
                    'max_volume_minute': max_volume,
                    'volume_distribution': {
                        'q25': float(hist['Volume'].quantile(0.25)),
                        'q50': float(hist['Volume'].quantile(0.50)),
                        'q75': float(hist['Volume'].quantile(0.75))
                    },
                    'last_price': float(hist['Close'].iloc[-1]),
                    'vwap': vwap,
                    'price_range': {
                        'high': float(hist['High'].max()),
                        'low': float(hist['Low'].min()),
                        'open': float(hist['Open'].iloc[0]) if len(hist) > 0 else 0
                    }
                },
                'metrics': {
                    'volume_trend': self._calculate_volume_trend(hist['Volume']),
                    'volume_spike_detected': self._detect_volume_spike(hist['Volume']),
                    'trading_intensity': self._calculate_trading_intensity(hist),
                    'volume_volatility': float(hist['Volume'].std() / hist['Volume'].mean() if hist['Volume'].mean() > 0 else 0)
                }
            }

            # Sauvegarde individuelle
            file_path = self.data_path / f'intraday_volume/{symbol}_volume_{timestamp}.json'
            with open(file_path, 'w') as f:
                json.dump(volume_data, f, indent=2, default=str)

            logger.info(f"📈 {symbol}: Volume={total_volume:,}, VWAP=${vwap:.2f}")

            # Respect du rate limit
            time.sleep(max(1, 60 / self.rate_limits['yahoo_finance']))
            return volume_data
            
        except Exception as e:
            logger.error(f"❌ Erreur collecte volume {symbol}: {e}")
            return None

    def detect_unusual_volume(self) -> List[Dict[str, Any]]:
        """Détecte les volumes anormaux (>200% de la moyenne 20 jours)"""
        logger.info("🔍 Détection des volumes anormaux...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unusual_volumes = []
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='30d')
                
                if len(hist) < 20:
                    logger.warning(f"⚠️ Données insuffisantes pour {symbol}")
                    continue
                    
                current_volume = hist['Volume'].iloc[-1]
                avg_volume_20d = hist['Volume'].iloc[-20:].mean()
                volume_ratio = current_volume / avg_volume_20d if avg_volume_20d > 0 else 0
                
                # Seuil de détection: volume > 200% de la moyenne
                if volume_ratio > 2.0:
                    price_change = float((hist['Close'].iloc[-1] / hist['Close'].iloc[-2] - 1) * 100) if len(hist) >= 2 else 0
                    
                    unusual_data = {
                        'symbol': symbol,
                        'timestamp': timestamp,
                        'current_volume': int(current_volume),
                        'avg_volume_20d': int(avg_volume_20d),
                        'volume_ratio': float(volume_ratio),
                        'price_change': price_change,
                        'alert_level': 'HIGH' if volume_ratio > 5.0 else 'MEDIUM',
                        'current_price': float(hist['Close'].iloc[-1]),
                        'analysis': {
                            'volume_percentile': float(hist['Volume'].rank(pct=True).iloc[-1] * 100),
                            'consecutive_high_volume_days': self._count_consecutive_high_volume(hist['Volume']),
                            'volume_trend_5d': self._calculate_volume_trend(hist['Volume'].tail(5))
                        }
                    }
                    
                    unusual_volumes.append(unusual_data)
                    logger.warning(f"🚨 VOLUME ANORMAL: {symbol} - {volume_ratio:.1f}x la moyenne ({current_volume:,} vs {avg_volume_20d:,.0f})")
                
                # Respect du rate limit
                time.sleep(max(1, 60 / self.rate_limits['yahoo_finance']))
                
            except Exception as e:
                logger.error(f"❌ Erreur détection volume anormal {symbol}: {e}")
        
        # Sauvegarde des alertes
        if unusual_volumes:
            alert_file = self.data_path / f'unusual_volume/unusual_volume_alert_{timestamp}.json'
            with open(alert_file, 'w') as f:
                json.dump(unusual_volumes, f, indent=2, default=str)
            
            logger.info(f"🚨 {len(unusual_volumes)} alertes de volume anormal générées")
            
        return unusual_volumes
    
    def analyze_volume_profile(self) -> None:
        """Analyse le profil de volume (distribution par niveau de prix)"""
        logger.info("📊 Analyse des profils de volume...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Limite à 3 symboles pour éviter rate limiting
        for symbol in self.symbols[:3]:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='5d', interval='1m')
                
                if hist.empty or len(hist) < 100:
                    logger.warning(f"⚠️ Données insuffisantes pour profil volume {symbol}")
                    continue
                
                # Nettoyage des données
                hist = hist.dropna()
                
                # Calcul du Volume Profile (VWAP par tranche de prix)
                try:
                    price_ranges = pd.cut(hist['Close'], bins=20, duplicates='drop')
                    volume_profile = hist.groupby(price_ranges, observed=True).agg({
                        'Volume': 'sum',
                        'Close': 'mean'
                    }).reset_index()
                    
                    if volume_profile.empty:
                        continue
                    
                    # Point of Control (POC) - niveau de prix avec le plus de volume
                    poc_idx = volume_profile['Volume'].idxmax()
                    poc_price = float(volume_profile.loc[poc_idx, 'Close'])
                    
                    profile_data = {
                        'symbol': symbol,
                        'timestamp': timestamp,
                        'profile': {
                            'poc_price': poc_price,
                            'total_volume': int(hist['Volume'].sum()),
                            'price_levels': len(volume_profile),
                            'value_area': self._calculate_value_area(volume_profile),
                            'volume_distribution': {
                                'high_volume_node': float(volume_profile['Volume'].max()),
                                'low_volume_node': float(volume_profile['Volume'].min()),
                                'avg_volume_per_level': float(volume_profile['Volume'].mean()),
                                'volume_concentration': float(volume_profile['Volume'].std() / volume_profile['Volume'].mean())
                            },
                            'price_statistics': {
                                'vwap': float((hist['Close'] * hist['Volume']).sum() / hist['Volume'].sum()),
                                'price_range': float(hist['High'].max() - hist['Low'].min()),
                                'current_price': float(hist['Close'].iloc[-1])
                            }
                        }
                    }
                    
                    file_path = self.data_path / f'volume_profile/{symbol}_profile_{timestamp}.json'
                    with open(file_path, 'w') as f:
                        json.dump(profile_data, f, indent=2, default=str)
                    
                    logger.info(f"📊 Profil volume {symbol}: POC à ${poc_price:.2f}, Volume total: {profile_data['profile']['total_volume']:,}")
                    
                except Exception as e:
                    logger.error(f"❌ Erreur calcul profil volume {symbol}: {e}")
                    continue
                
                # Respect du rate limit
                time.sleep(max(2, 60 / self.rate_limits['yahoo_finance']))
                
            except Exception as e:
                logger.error(f"❌ Erreur analyse profil volume {symbol}: {e}")

    def collect_institutional_volume(self) -> None:
        """Collecte les données de volume institutionnel via l'API Polygon"""
        polygon_key = self.api_keys.get('polygon')
        if not polygon_key:
            logger.warning("⚠️ Clé API Polygon manquante pour volume institutionnel")
            return

        logger.info("🏛️ Collecte du volume institutionnel...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = self.data_path / 'institutional_volume'

        base_url = "https://api.polygon.io/v3/trades"

        for symbol in self.symbols[:3]:  # Limitation pour respecter les quotas
            try:
                url = f"{base_url}/{symbol.upper()}"
                params = {
                    'apikey': polygon_key,
                    'limit': 1000,
                    'timestamp.gte': (datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d')
                }

                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                trades = data.get("results", [])
                if not trades:
                    logger.warning(f"⚠️ Aucune donnée de trade pour {symbol}")
                    continue

                # Classification des trades
                large_trades = [t for t in trades if t.get("s", 0) >= 10_000]  # >= 10k shares
                block_trades = [t for t in trades if t.get("s", 0) >= 100_000]  # >= 100k shares
                
                # Calcul des métriques
                total_large_volume = sum(t.get("s", 0) for t in large_trades)
                total_block_volume = sum(t.get("s", 0) for t in block_trades)
                avg_trade_size = np.mean([t.get("s", 0) for t in trades]) if trades else 0

                institutional_data = {
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'source': 'polygon',
                    'summary': {
                        'total_trades': len(trades),
                        'large_trades_count': len(large_trades),
                        'block_trades_count': len(block_trades),
                        'total_large_volume': int(total_large_volume),
                        'total_block_volume': int(total_block_volume),
                        'avg_trade_size': float(avg_trade_size),
                        'institutional_percentage': float((total_large_volume + total_block_volume) / sum(t.get("s", 0) for t in trades) * 100) if trades else 0
                    },
                    'large_trades': large_trades[:50],  # Limite pour la taille du fichier
                    'block_trades': block_trades[:20],
                    'analysis': {
                        'institutional_flow': self._analyze_institutional_flow(large_trades, block_trades),
                        'dark_pool_estimate': None,  # Non disponible directement
                        'trade_size_distribution': self._calculate_trade_size_distribution(trades)
                    }
                }

                file_path = output_dir / f'{symbol}_institutional_{timestamp}.json'
                with open(file_path, 'w') as f:
                    json.dump(institutional_data, f, indent=2, default=str)

                logger.info(f"✅ Volume institutionnel {symbol}: {len(large_trades)} gros trades, {len(block_trades)} block trades")

                # Respect du rate limit Polygon
                time.sleep(max(1, 60 / self.rate_limits.get('polygon', 100)))

            except requests.exceptions.RequestException as e:
                logger.error(f"⛔ Erreur HTTP pour {symbol}: {e}")
            except Exception as e:
                logger.error(f"❌ Erreur volume institutionnel {symbol}: {e}")

    def _calculate_volume_trend(self, volume_series: pd.Series) -> str:
        """Calcule la tendance du volume (hausse/baisse)"""
        if len(volume_series) < 10:
            return 'insufficient_data'
            
        recent_avg = volume_series.tail(min(10, len(volume_series))).mean()
        
        if len(volume_series) >= 20:
            previous_avg = volume_series.iloc[-20:-10].mean()
        else:
            previous_avg = volume_series.head(min(10, len(volume_series) // 2)).mean()
        
        if previous_avg == 0:
            return 'stable'
            
        if recent_avg > previous_avg * 1.2:
            return 'increasing'
        elif recent_avg < previous_avg * 0.8:
            return 'decreasing'
        else:
            return 'stable'
    
    def _detect_volume_spike(self, volume_series: pd.Series) -> bool:
        """Détecte les pics de volume"""
        if len(volume_series) < 5:
            return False
            
        current_volume = volume_series.iloc[-1]
        avg_recent = volume_series.tail(min(10, len(volume_series))).mean()
        
        return current_volume > avg_recent * 3
    
    def _calculate_trading_intensity(self, hist_data: pd.DataFrame) -> float:
        """Calcule l'intensité des échanges"""
        if hist_data.empty or hist_data['Volume'].mean() == 0:
            return 0.0
            
        volume_velocity = hist_data['Volume'].std() / hist_data['Volume'].mean()
        return min(float(volume_velocity * 100), 100.0)  # Normalisé sur 100
    
    def _count_consecutive_high_volume(self, volume_series: pd.Series) -> int:
        """Compte les jours consécutifs de volume élevé"""
        if len(volume_series) < 2:
            return 0
            
        avg_volume = volume_series.mean()
        high_volume_threshold = avg_volume * 1.5
        
        consecutive_days = 0
        for volume in reversed(volume_series.tolist()):
            if volume > high_volume_threshold:
                consecutive_days += 1
            else:
                break
                
        return consecutive_days
    
    def _calculate_value_area(self, volume_profile: pd.DataFrame) -> Dict[str, float]:
        """Calcule la zone de valeur (70% du volume)"""
        if volume_profile.empty:
            return {'min_price': 0.0, 'max_price': 0.0, 'volume_percentage': 0.0}
            
        total_volume = volume_profile['Volume'].sum()
        if total_volume == 0:
            return {'min_price': 0.0, 'max_price': 0.0, 'volume_percentage': 0.0}
            
        target_volume = total_volume * 0.7
        
        # Tri par volume décroissant
        sorted_profile = volume_profile.sort_values('Volume', ascending=False).copy()
        cumulative_volume = 0
        value_area_levels = []
        
        for _, row in sorted_profile.iterrows():
            cumulative_volume += row['Volume']
            value_area_levels.append(float(row['Close']))
            
            if cumulative_volume >= target_volume:
                break
        
        if not value_area_levels:
            return {'min_price': 0.0, 'max_price': 0.0, 'volume_percentage': 0.0}
        
        return {
            'min_price': min(value_area_levels),
            'max_price': max(value_area_levels),
            'volume_percentage': float((cumulative_volume / total_volume) * 100)
        }
    
    def _analyze_institutional_flow(self, large_trades: List[Dict], block_trades: List[Dict]) -> str:
        """Analyse le flux institutionnel (achat/vente)"""
        if not large_trades and not block_trades:
            return 'neutral'
        
        # Analyse simplifiée basée sur la taille des trades
        # Dans un vrai système, on analyserait les conditions d'échange
        total_large_volume = sum(t.get('s', 0) for t in large_trades)
        total_block_volume = sum(t.get('s', 0) for t in block_trades)
        
        if total_block_volume > total_large_volume * 2:
            return 'strong_institutional'
        elif total_large_volume > 0:
            return 'moderate_institutional'
        else:
            return 'neutral'
    
    def _calculate_trade_size_distribution(self, trades: List[Dict]) -> Dict[str, int]:
        """Calcule la distribution des tailles de trades"""
        if not trades:
            return {'small': 0, 'medium': 0, 'large': 0, 'block': 0}
        
        distribution = {'small': 0, 'medium': 0, 'large': 0, 'block': 0}
        
        for trade in trades:
            size = trade.get('s', 0)
            if size < 1000:
                distribution['small'] += 1
            elif size < 10000:
                distribution['medium'] += 1
            elif size < 100000:
                distribution['large'] += 1
            else:
                distribution['block'] += 1
        
        return distribution
    
    def generate_volume_report(self) -> Dict[str, Any]:
        """Génère un rapport consolidé des volumes"""
        logger.info("📋 Génération du rapport de volume...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Collecte des données récentes pour le rapport
        recent_data = []
        try:
            for symbol in self.symbols[:5]:  # Limite pour la performance
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1d')
                if not hist.empty:
                    current_volume = int(hist['Volume'].iloc[-1])
                    current_price = float(hist['Close'].iloc[-1])
                    recent_data.append({
                        'symbol': symbol,
                        'volume': current_volume,
                        'price': current_price
                    })
        except Exception as e:
            logger.error(f"❌ Erreur collecte données pour rapport: {e}")
        
        # Tri par volume décroissant
        recent_data.sort(key=lambda x: x['volume'], reverse=True)
        
        report = {
            'timestamp': timestamp,
            'generation_time': datetime.now().isoformat(),
            'market_summary': {
                'total_symbols_monitored': len(self.symbols),
                'symbols_with_data': len(recent_data),
                'collection_status': 'active',
                'last_update': datetime.now().isoformat(),
                'data_sources': ['yahoo_finance', 'polygon', 'finnhub']
            },
            'top_volume_symbols': recent_data[:10],
            'volume_alerts': {
                'unusual_volume_detected': 0,  # À calculer depuis les fichiers récents
                'high_activity_symbols': [s['symbol'] for s in recent_data[:3]]
            },
            'market_sentiment': 'neutral',  # À améliorer avec analyse sentiment
            'statistics': {
                'avg_volume': np.mean([s['volume'] for s in recent_data]) if recent_data else 0,
                'total_volume_monitored': sum(s['volume'] for s in recent_data),
                'price_range': {
                    'min': min(s['price'] for s in recent_data) if recent_data else 0,
                    'max': max(s['price'] for s in recent_data) if recent_data else 0
                }
            }
        }
        
        # Sauvegarde du rapport
        report_file = self.data_path / f'volume_analysis/volume_report_{timestamp}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📋 Rapport de volume généré: {len(recent_data)} symboles analysés")
        return report
    
    def run_collection_cycle(self) -> None:
        """Exécute un cycle complet de collecte de volume"""
        logger.info("🔄 Début du cycle de collecte de volume")
        
        try:
            start_time = time.time()
            
            # Collecte principale (toujours)
            self.collect_intraday_volume()
            time.sleep(5)
            
            # Détection d'anomalies (toujours)
            unusual_vols = self.detect_unusual_volume()
            time.sleep(5)
            
            current_time = datetime.now()
            
            # Analyse des profils (toutes les 15 minutes)
            if current_time.minute % 15 == 0:
                self.analyze_volume_profile()
                time.sleep(5)
            
            # Volume institutionnel (toutes les heures)
            if current_time.minute == 0 and self.api_keys.get('polygon'):
                self.collect_institutional_volume()
                time.sleep(5)
            
            # Rapport (toutes les 30 minutes)
            if current_time.minute % 30 == 0:
                self.generate_volume_report()
            
            cycle_time = time.time() - start_time
            logger.info(f"✅ Cycle terminé en {cycle_time:.2f}s - {len(unusual_vols)} alertes générées")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du cycle de collecte: {e}")
            raise
    
    def start_continuous_collection(self) -> None:
        """Démarre la collecte continue de volume"""
        logger.info("🚀 Démarrage du Volume Collector en mode continu")
        logger.info(f"📊 Symboles surveillés: {', '.join(self.symbols)}")
        
        interval = int(os.environ.get('COLLECTION_INTERVAL', 300))  # 5 minutes par défaut
        logger.info(f"⏱️ Intervalle de collecte: {interval} secondes")
        
        while True:
            try:
                self.run_collection_cycle()
                
                logger.info(f"⏱️ Prochaine collectedans {interval} secondes...")
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("⏹️ Arrêt demandé par l'utilisateur")
                break
            except Exception as e:
                logger.error(f"❌ Erreur critique: {e}")
                logger.info(f"⏱️ Reprise dans {interval} secondes...")
                time.sleep(interval)

