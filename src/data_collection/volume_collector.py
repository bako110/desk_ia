"""
NASDAQ IA Trading - Volume Collector (Version Optimisée)
Collecte les données de volume de trading en temps réel
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
from concurrent.futures import ThreadPoolExecutor
from alpha_vantage.timeseries import TimeSeries
import finnhub
import yaml
from typing import Dict, List, Optional, Any

# Configuration du logging simplifié
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/volume_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('VolumeCollector')

class VolumeCollector:
    """Collecteur optimisé pour les données de volume de trading"""

    @staticmethod
    def load_api_keys(yaml_path: str = "config/api_key.yaml") -> Dict[str, str]:
        """Charge les clés API depuis le fichier YAML"""
        try:
            config_path = Path(yaml_path)
            if not config_path.exists():
                logger.warning(f"⚠️ Fichier de configuration {yaml_path} introuvable")
                return {}
                
            with open(config_path, 'r', encoding='utf-8') as f:
                keys = yaml.safe_load(f)
                logger.info("🔑 Clés API chargées avec succès")
                return keys or {}
        except FileNotFoundError:
            logger.warning(f"⚠️ Fichier {yaml_path} non trouvé")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"❌ Erreur format YAML: {e}")
            return {}
        except Exception as e:
            logger.error(f"❌ Erreur chargement des clés API: {e}")
            return {}

    def __init__(self, symbols: Optional[List[str]] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialise le collecteur de volumes
        
        Args:
            symbols: Liste des symboles à surveiller (optionnel)
            config: Configuration personnalisée (optionnel)
        """
        # Configuration de base
        self.data_path = Path(os.environ.get('DATA_PATH', '/app/data/volumes'))
        
        # Symboles - priorité aux paramètres, puis variables d'environnement
        if symbols and isinstance(symbols, list):
            self.symbols = [s.strip().upper() for s in symbols]
        elif isinstance(symbols, str):
            # Si symbols est une chaîne, la diviser
            self.symbols = [s.strip().upper() for s in symbols.split(',')]
        else:
            # Utiliser les variables d'environnement
            symbols_env = os.environ.get('SYMBOLS', 'AAPL,GOOGL,MSFT,TSLA,NVDA')
            self.symbols = [s.strip().upper() for s in symbols_env.split(',')]
        
        # Configuration personnalisée ou par défaut
        self.config = config or {}
        
        # Configuration des APIs
        self.api_keys = self.load_api_keys()
        
        # Rate limits avec configuration personnalisée
        default_rates = {
            'alphavantage': 5,
            'polygon': 100,
            'finnhub': 60,
            'yahoo_finance': 120
        }
        
        self.rate_limits = {}
        for api, default_rate in default_rates.items():
            # Priorité : config personnalisée > variable d'env > défaut
            self.rate_limits[api] = (
                self.config.get('rate_limits', {}).get(api) or
                int(os.environ.get(f'{api.upper()}_RATE_LIMIT', default_rate))
            )
        
        # Création des répertoires avec gestion d'erreurs
        try:
            self._setup_directories()
        except Exception as e:
            logger.error(f"❌ Erreur création répertoires: {e}")
            
        # Initialisation des clients API avec gestion d'erreurs
        try:
            self._initialize_clients()
        except Exception as e:
            logger.error(f"❌ Erreur initialisation clients: {e}")
            
        logger.info(f"📊 Volume Collector initialisé - {len(self.symbols)} symboles")
        logger.info(f"📁 Répertoire de données: {self.data_path}")
        logger.info(f"🔧 Sources API disponibles: {list(self.api_keys.keys())}")

    def _setup_directories(self):
        """Crée la structure des répertoires"""
        try:
            directories = ['intraday', 'unusual', 'reports', 'alphavantage', 'polygon', 'finnhub']
            for directory in directories:
                dir_path = self.data_path / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"📁 Répertoire créé/vérifié: {dir_path}")
        except Exception as e:
            logger.error(f"❌ Erreur création répertoires: {e}")
            # Créer un répertoire de base minimal
            self.data_path.mkdir(parents=True, exist_ok=True)

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

        except Exception as e:
            logger.error(f"❌ Erreur initialisation clients API: {e}")

    def collect_volumes(self) -> List[Dict[str, Any]]:
        """Collecte les volumes pour tous les symboles"""
        logger.info("📊 Collecte des volumes...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._get_symbol_data, symbol): symbol
                for symbol in self.symbols
            }
            
            for future in futures:
                try:
                    result = future.result(timeout=30)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"❌ Erreur: {e}")

        # Sauvegarde consolidée
        if results:
            file_path = self.data_path / f'intraday/volumes_{timestamp}.json'
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
        logger.info(f"✅ {len(results)} symboles collectés")
        return results

    def _get_symbol_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Collecte les données pour un symbole depuis plusieurs sources"""
        try:
            # Yahoo Finance (source principale)
            ticker = yf.Ticker(symbol)
            hist_1m = ticker.history(period='1d', interval='1m').dropna()
            hist_30d = ticker.history(period='30d').dropna()
            
            if hist_1m.empty or hist_30d.empty:
                return None

            # Métriques de base Yahoo Finance
            current_volume = int(hist_1m['Volume'].iloc[-1])
            total_volume_today = int(hist_1m['Volume'].sum())
            avg_volume_20d = int(hist_30d['Volume'].tail(20).mean())
            current_price = float(hist_1m['Close'].iloc[-1])
            vwap = float((hist_1m['Close'] * hist_1m['Volume']).sum() / hist_1m['Volume'].sum())
            
            # Données de base
            data = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'sources': ['yahoo_finance'],
                'yahoo_finance': {
                    'current_volume': current_volume,
                    'total_volume_today': total_volume_today,
                    'avg_volume_20d': avg_volume_20d,
                    'current_price': current_price,
                    'vwap': round(vwap, 2)
                }
            }
            
            # Alpha Vantage - Données complémentaires
            if self.av_client:
                try:
                    av_data, _ = self.av_client.get_intraday(symbol, interval='1min', outputsize='compact')
                    if av_data:
                        latest_av = list(av_data.keys())[0]
                        av_volume = int(av_data[latest_av]['5. volume'])
                        data['sources'].append('alphavantage')
                        data['alphavantage'] = {
                            'volume': av_volume,
                            'price': float(av_data[latest_av]['4. close'])
                        }
                        time.sleep(60 / self.rate_limits['alphavantage'])
                except Exception as e:
                    logger.warning(f"⚠️ Alpha Vantage error for {symbol}: {e}")
            
            # Finnhub - Données en temps réel
            if self.finnhub_client:
                try:
                    quote = self.finnhub_client.quote(symbol)
                    if quote:
                        data['sources'].append('finnhub')
                        data['finnhub'] = {
                            'current_price': float(quote.get('c', 0)),
                            'change': float(quote.get('d', 0)),
                            'change_percent': float(quote.get('dp', 0))
                        }
                        time.sleep(60 / self.rate_limits['finnhub'])
                except Exception as e:
                    logger.warning(f"⚠️ Finnhub error for {symbol}: {e}")
            
            # Polygon - Volume institutionnel
            if self.api_keys.get('polygon'):
                try:
                    polygon_data = self._get_polygon_data(symbol)
                    if polygon_data:
                        data['sources'].append('polygon')
                        data['polygon'] = polygon_data
                except Exception as e:
                    logger.warning(f"⚠️ Polygon error for {symbol}: {e}")
            
            # Calculs finaux
            volume_ratio = current_volume / avg_volume_20d if avg_volume_20d > 0 else 0
            data.update({
                'volume_ratio': round(volume_ratio, 2),
                'is_unusual_volume': volume_ratio > 2.0,
                'alert_level': 'HIGH' if volume_ratio > 5.0 else 'MEDIUM' if volume_ratio > 2.0 else 'NORMAL'
            })
            
            logger.info(f"📈 {symbol}: Vol={current_volume:,} ({volume_ratio:.1f}x), Sources={len(data['sources'])}")
            
            # Rate limit Yahoo Finance
            time.sleep(60 / self.rate_limits['yahoo_finance'])
            return data
            
        except Exception as e:
            logger.error(f"❌ Erreur {symbol}: {e}")
            return None

    def _get_polygon_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Collecte les données Polygon pour un symbole"""
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"
            params = {'apikey': self.api_keys['polygon']}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('results'):
                result = data['results'][0]
                time.sleep(60 / self.rate_limits['polygon'])
                return {
                    'volume': int(result.get('v', 0)),
                    'open': float(result.get('o', 0)),
                    'close': float(result.get('c', 0)),
                    'high': float(result.get('h', 0)),
                    'low': float(result.get('l', 0))
                }
        except Exception as e:
            logger.warning(f"⚠️ Polygon API error: {e}")
        return None

    def save_api_data(self, data_list: List[Dict[str, Any]]):
        """Sauvegarde les données par source API"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Séparation par source
        for item in data_list:
            symbol = item['symbol']
            sources = item.get('sources', [])
            
            # Sauvegarde Alpha Vantage
            if 'alphavantage' in sources and 'alphavantage' in item:
                av_file = self.data_path / f'alphavantage/{symbol}_av_{timestamp}.json'
                with open(av_file, 'w') as f:
                    json.dump(item['alphavantage'], f, indent=2)
            
            # Sauvegarde Finnhub
            if 'finnhub' in sources and 'finnhub' in item:
                fh_file = self.data_path / f'finnhub/{symbol}_fh_{timestamp}.json'
                with open(fh_file, 'w') as f:
                    json.dump(item['finnhub'], f, indent=2)
            
            # Sauvegarde Polygon
            if 'polygon' in sources and 'polygon' in item:
                pg_file = self.data_path / f'polygon/{symbol}_pg_{timestamp}.json'
                with open(pg_file, 'w') as f:
                    json.dump(item['polygon'], f, indent=2)

    def detect_unusual_volumes(self) -> List[Dict[str, Any]]:
        """Détecte et sauvegarde les volumes anormaux"""
        logger.info("🔍 Détection volumes anormaux...")
        
        # Collecte les données
        all_data = self.collect_volumes()
        
        # Sauvegarde par API
        self.save_api_data(all_data)
        
        unusual_volumes = [d for d in all_data if d.get('is_unusual_volume', False)]
        
        if unusual_volumes:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = self.data_path / f'unusual/alert_{timestamp}.json'
            
            with open(file_path, 'w') as f:
                json.dump(unusual_volumes, f, indent=2, default=str)
            
            for vol in unusual_volumes:
                logger.warning(f"🚨 ALERTE: {vol['symbol']} - {vol['volume_ratio']}x la moyenne")
        
        return unusual_volumes

    def generate_report(self) -> Dict[str, Any]:
        """Génère un rapport de synthèse"""
        logger.info("📋 Génération du rapport...")
        
        # Collecte des données actuelles
        current_data = self.collect_volumes()
        
        if not current_data:
            return {}
        
        # Tri par volume décroissant (Yahoo Finance comme référence)
        current_data.sort(key=lambda x: x.get('yahoo_finance', {}).get('current_volume', 0), reverse=True)
        
        # Statistiques globales
        total_volume = sum(d.get('yahoo_finance', {}).get('current_volume', 0) for d in current_data)
        avg_volume = total_volume / len(current_data) if current_data else 0
        unusual_count = len([d for d in current_data if d['is_unusual_volume']])
        
        # Compilation des sources utilisées
        all_sources = set()
        for d in current_data:
            all_sources.update(d.get('sources', []))
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'symbols_monitored': len(self.symbols),
                'symbols_active': len(current_data),
                'total_volume': total_volume,
                'avg_volume': int(avg_volume),
                'unusual_volume_alerts': unusual_count,
                'data_sources_active': list(all_sources)
            },
            'top_volumes': current_data[:5],
            'unusual_volumes': [d for d in current_data if d['is_unusual_volume']],
            'market_activity': 'HIGH' if unusual_count > 2 else 'MEDIUM' if unusual_count > 0 else 'NORMAL'
        }
        
        # Sauvegarde
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = self.data_path / f'reports/report_{timestamp}.json'
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📋 Rapport généré - {unusual_count} alertes")
        return report

    def run_cycle(self):
        """Exécute un cycle de collecte complet"""
        logger.info("🔄 Cycle de collecte...")
        
        try:
            start_time = time.time()
            
            # Détection des volumes anormaux
            unusual_volumes = self.detect_unusual_volumes()
            
            # Génération du rapport (toutes les 30 minutes)
            current_time = datetime.now()
            if current_time.minute % 30 == 0:
                self.generate_report()
            
            cycle_time = time.time() - start_time
            logger.info(f"✅ Cycle terminé en {cycle_time:.1f}s - {len(unusual_volumes)} alertes")
            
        except Exception as e:
            logger.error(f"❌ Erreur cycle: {e}")

    def start_monitoring(self):
        """Démarre la surveillance continue"""
        logger.info("🚀 Démarrage de la surveillance continue")
        logger.info(f"📊 Symboles: {', '.join(self.symbols)}")
        
        interval = int(os.environ.get('COLLECTION_INTERVAL', 300))  # 5 minutes
        logger.info(f"⏱️ Intervalle: {interval}s")
        
        while True:
            try:
                self.run_cycle()
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("⏹️ Arrêt demandé")
                break
            except Exception as e:
                logger.error(f"❌ Erreur: {e}")
                time.sleep(interval)
