# =============================================================================
# CLASSE DE BASE POUR TOUS LES COLLECTEURS - NASDAQ IA TRADING
# =============================================================================

import pandas as pd
import numpy as np
import requests
import time
import logging
import sqlite3
import yaml
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION DES CLÉS API
# =============================================================================

@dataclass
class APIConfig:
    price_apis: Optional[Dict[str, str]] = None
    fundamental_apis: Optional[Dict[str, str]] = None
    sentiment_apis: Optional[Dict[str, str]] = None
    macro_apis: Optional[Dict[str, str]] = None
    alternative_apis: Optional[Dict[str, str]] = None
    rate_limits: Optional[Dict[str, int]] = None

def load_api_config_from_yaml(path: str = "config/apikey.yaml") -> APIConfig:
    try:
        with open(path, 'r') as file:
            config_dict = yaml.safe_load(file)
            return APIConfig(**config_dict)
    except Exception as e:
        print(f"[ERREUR] Chargement config YAML échoué : {e}")
        return APIConfig()

# =============================================================================
# CLASSE DE BASE DES COLLECTEURS
# =============================================================================

class BaseCollector(ABC):
    """Classe de base pour tous les collecteurs de données."""

    def __init__(self, api_config: APIConfig):
        self.api_config = api_config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session = requests.Session()
        self.rate_limit_delay = 1.0  # Délai de base entre les requêtes API
        self._setup_logging()

    def _setup_logging(self):
        """Configuration du logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/data_collection.log'),
                logging.StreamHandler()
            ]
        )

    @abstractmethod
    def collect_data(self, symbols: List[str], **kwargs) -> pd.DataFrame:
        """Méthode abstraite à implémenter pour collecter les données."""
        pass

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Valide que les données sont exploitables."""
        if data.empty:
            self.logger.warning("Données vides reçues.")
            return False
        if data.isnull().sum().sum() > len(data) * 0.5:
            self.logger.warning("Trop de valeurs manquantes dans les données.")
            return False
        return True

    def save_to_database(self, data: pd.DataFrame, table_name: str, db_path: str = "data/nasdaq_ai.db"):
        """Sauvegarde des données dans une base SQLite."""
        try:
            conn = sqlite3.connect(db_path)
            data.to_sql(table_name, conn, if_exists='append', index=False)
            conn.commit()
            conn.close()
            self.logger.info(f"Données sauvegardées dans la table {table_name}.")
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde dans la base : {e}")

    def save_to_csv(self, data: pd.DataFrame, file_path: str):
        """Sauvegarde des données au format CSV."""
        try:
            data.to_csv(file_path, index=False)
            self.logger.info(f"Données sauvegardées dans le fichier CSV : {file_path}")
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde CSV : {e}")

    def rate_limit(self):
        """Pause entre les appels API pour respecter les limitations."""
        time.sleep(self.rate_limit_delay)

    def _safe_float(self, value) -> Optional[float]:
        """Conversion sécurisée en float."""
        try:
            return float(value) if value and value != 'None' else None
        except (ValueError, TypeError):
            return None

    def _safe_int(self, value) -> Optional[int]:
        """Conversion sécurisée en int."""
        try:
            return int(value) if value and value != 'None' else None
        except (ValueError, TypeError):
            return None

    def handle_api_error(self, response: requests.Response, source: str) -> bool:
        """Gestion des erreurs de réponse API."""
        if response.status_code == 200:
            return True
        elif response.status_code == 429:
            self.logger.warning(f"Rate limit atteint pour {source}. Attente de 60 secondes.")
            time.sleep(60)
            return False
        elif response.status_code == 401:
            self.logger.error(f"Clé API invalide pour {source}.")
            return False
        else:
            self.logger.error(f"Erreur {response.status_code} pour la source {source}.")
            return False
