# import logging
# from pathlib import Path
# import yaml
# import os

# from src.data_collection.price_collector import PriceCollector
# from src.data_collection.fundamental_collector import FundamentalCollector
# from src.data_collection.sentiment_collector import SentimentCollector
# from src.data_collection.macro_collector import MacroCollector
# from src.data_collection.volume_collector import VolumeCollector
# from src.data_collection.alternative_collector import AlternativeCollector
# from src.data_collection.options_collector import OptionsCollector
# from src.data_collection.real_time_collector import RealTimeCollector


# class CollectorManager:
#     def __init__(self):
#         self.logger = logging.getLogger(__name__)
#         self.base_path = Path("data")  # dossier data
#         self.config_path = Path("config") / "api_keys.yaml"  
#         self.config = self._load_config()
#         self._init_collectors()

#     def _load_config(self):
#         # Créer le dossier config s'il n'existe pas
#         self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
#         try:
#             with open(self.config_path, 'r', encoding='utf-8') as f:
#                 config = yaml.safe_load(f)
#             self.logger.info(f"Config loaded from {self.config_path}")
#             return config
#         except FileNotFoundError:
#             self.logger.warning(f"Fichier de configuration {self.config_path} non trouvé")
#             # Créer un fichier de configuration par défaut
#             self._create_default_config()
#             self.logger.info(f"Fichier de configuration par défaut créé: {self.config_path}")
#             # Charger la configuration par défaut
#             with open(self.config_path, 'r', encoding='utf-8') as f:
#                 config = yaml.safe_load(f)
#             return config
#         except yaml.YAMLError as e:
#             self.logger.error(f"Erreur lors du parsing YAML: {e}")
#             raise
#         except Exception as e:
#             self.logger.error(f"Erreur inattendue lors du chargement de la config: {e}")
#             raise

#     def _create_default_config(self):
#         """Crée un fichier de configuration par défaut avec des clés API vides"""
#         default_config = {
# 'api_keys': {
#     'alpha_vantage': 'RU6W0PWAUZ0JYD0A',
#     'yahoo_finance': 'TONE_KEY_YAHOO_FINANCE_ICI',
#     'polygon': '6_TgTH0XU8AdgToOqMqfrEsE4PkwRJda',
#     'iex_cloud': 'TONE_KEY_IEXCLOUD_ICI',
#     'quandl': 'CchiUoMKN9thkVHWm_pd',
#     'twitter': 'YOUR_TWITTER_API_KEY',
#     'reddit': 'YOUR_REDDIT_API_KEY',
#     'news_api': 'YOUR_NEWS_API_KEY',
#     'finnhub': 'd0ng2fpr01qi1cve64bgd0ng2fpr01qi1cve64c0',
#     'bloomberg': 'TONE_KEY_BLOOMBERG_ICI',
#     'refinitiv': 'TONE_KEY_REFINITIV_ICI',
#     'social_market_analytics': 'TONE_KEY_SMA_ICI',
#     'estimize': 'TONE_KEY_ESTIMIZE_ICI',
#     'fred': '5ab3bfadeb631202a8dad2a52fd01821',
#     'thinknum': 'TONE_KEY_THINKNUM_ICI',
#     'satellite_logic': 'TONE_KEY_SATELLITE_LOGIC_ICI'
# },

#             'settings': {
#                 'timeout': 30,
#                 'retry_count': 3,
#                 'delay_between_requests': 1
#             }
#         }
        
#         try:
#             with open(self.config_path, 'w', encoding='utf-8') as f:
#                 yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True, indent=2)
#         except Exception as e:
#             self.logger.error(f"Erreur lors de la création du fichier de config par défaut: {e}")
#             raise

#     def _init_collectors(self):
#         # Créer les dossiers de données s'ils n'existent pas
#         self.base_path.mkdir(parents=True, exist_ok=True)
        
#         # Passer les paramètres dans le bon ordre : data_path d'abord, config ensuite
#         try:
#             self.price_collector = PriceCollector(self.base_path / "price_data", self.config)
            
#             # CORRECTION: Passer data_path et config séparément
#             self.fundamental_collector = FundamentalCollector(
#                 data_path=str(self.base_path / "fundamental"), 
#                 config=self.config
#             )
            
#             self.sentiment_collector = SentimentCollector(self.config)
#             self.macro_collector = MacroCollector(self.base_path / "macro", self.config)
#             self.volume_collector = VolumeCollector(self.base_path / "volume_flow", self.config)
#             self.alternative_collector = AlternativeCollector(self.base_path / "alternative", self.config)
#             self.options_collector = OptionsCollector(self.base_path / "price_data/options", self.config)
#             self.real_time_collector = RealTimeCollector(self.base_path / "real_time", self.config)
#             self.logger.info("Tous les collecteurs initialisés avec succès")
#         except Exception as e:
#             self.logger.error(f"Erreur lors de l'initialisation des collecteurs: {e}")
#             raise

#     def collect_all(self):
#         self.logger.info("Début de la collecte de toutes les données...")

#         # Liste des collecteurs avec leurs noms pour un traitement uniforme
#         collectors = [
#             (self.price_collector, "prix"),
#             (self.fundamental_collector, "fondamentaux"),
#             (self.sentiment_collector, "sentiment"),
#             (self.macro_collector, "macro"),
#             (self.volume_collector, "volume"),
#             (self.alternative_collector, "alternative"),
#             (self.options_collector, "options"),
#             (self.real_time_collector, "temps réel")
#         ]

#         success_count = 0
#         total_count = len(collectors)

#         for collector, name in collectors:
#             try:
#                 collector.collect()
#                 self.logger.info(f"Collecte des données {name} terminée avec succès.")
#                 success_count += 1
#             except Exception as e:
#                 self.logger.error(f"Erreur collecte {name}: {e}")

#         self.logger.info(f"Collecte complète terminée. {success_count}/{total_count} collecteurs ont réussi.")

#     def validate_config(self):
#         """Valide que la configuration contient les clés API nécessaires"""
#         if not self.config:
#             return False
        
#         api_keys = self.config.get('api_keys', {})
#         missing_keys = []
        
#         for key, value in api_keys.items():
#             if not value or value.startswith('YOUR_'):
#                 missing_keys.append(key)
        
#         if missing_keys:
#             self.logger.warning(f"Clés API manquantes ou non configurées: {missing_keys}")
#             self.logger.warning("Veuillez modifier le fichier config/apikey.yaml avec vos vraies clés API")
#             return False
        
#         return True


# if __name__ == "__main__":
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.StreamHandler(),
#             logging.FileHandler('collector.log', encoding='utf-8')
#         ]
#     )
    
#     try:
#         manager = CollectorManager()
        
#         # Valider la configuration
#         if not manager.validate_config():
#             print("\n⚠️ ATTENTION: Certaines clés API ne sont pas configurées.")
#             print("Veuillez modifier le fichier 'config/apikey.yaml' avec vos vraies clés API.")
#             print("Le fichier a été créé avec des valeurs par défaut.\n")
        
#         manager.collect_all()
        
#     except Exception as e:
#         logging.error(f"Erreur fatale: {e}")
#         print(f"\n❌ Erreur fatale: {e}")
#         print("Vérifiez les logs pour plus de détails.")

import logging
from pathlib import Path
import yaml
import os

from src.data_collection.price_collector import PriceCollector
from src.data_collection.fundamental_collector import FundamentalCollector
from src.data_collection.sentiment_collector import SentimentCollector
# from src.data_collection.macro_collector import MacroCollector
# from src.data_collection.volume_collector import VolumeCollector
# from src.data_collection.alternative_collector import AlternativeCollector
# from src.data_collection.options_collector import OptionsCollector
# from src.data_collection.real_time_collector import RealTimeCollector


class CollectorManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_path = Path("data")
        self.config_path = Path("config") / "api_keys.yaml"
        self.config = self._load_config()
        self._init_collectors()

    def _load_config(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Config loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            self.logger.warning(f"Fichier de configuration {self.config_path} non trouvé")
            self._create_default_config()
            self.logger.info(f"Fichier de configuration par défaut créé: {self.config_path}")
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            self.logger.error(f"Erreur lors du parsing YAML: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Erreur inattendue lors du chargement de la config: {e}")
            raise

    def _create_default_config(self):
        default_config = {
            'api_keys': {
                'alpha_vantage': 'RU6W0PWAUZ0JYD0A',
                'yahoo_finance': 'TONE_KEY_YAHOO_FINANCE_ICI',
                'polygon': '6_TgTH0XU8AdgToOqMqfrEsE4PkwRJda',
                'iex_cloud': 'TONE_KEY_IEXCLOUD_ICI',
                'quandl': 'CchiUoMKN9thkVHWm_pd',
                'twitter': 'YOUR_TWITTER_API_KEY',
                'reddit': 'YOUR_REDDIT_API_KEY',
                'news_api': 'YOUR_NEWS_API_KEY',
                'finnhub': 'd0ng2fpr01qi1cve64bgd0ng2fpr01qi1cve64c0',
                'bloomberg': 'TONE_KEY_BLOOMBERG_ICI',
                'refinitiv': 'TONE_KEY_REFINITIV_ICI',
                'social_market_analytics': 'TONE_KEY_SMA_ICI',
                'estimize': 'TONE_KEY_ESTIMIZE_ICI',
                'fred': '5ab3bfadeb631202a8dad2a52fd01821',
                'thinknum': 'TONE_KEY_THINKNUM_ICI',
                'satellite_logic': 'TONE_KEY_SATELLITE_LOGIC_ICI'
            },
            'settings': {
                'timeout': 30,
                'retry_count': 3,
                'delay_between_requests': 1
            }
        }

        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True, indent=2)
        except Exception as e:
            self.logger.error(f"Erreur lors de la création du fichier de config par défaut: {e}")
            raise

    def _init_collectors(self):
        self.base_path.mkdir(parents=True, exist_ok=True)

        try:
            self.price_collector = PriceCollector(self.base_path / "price_data", self.config)
            self.logger.info("Collecteur de prix initialisé avec succès")

            self.fundamental_collector = FundamentalCollector(self.base_path / "fundamental", self.config)
            self.logger.info("Collecteur fondamental initialisé avec succès")

            self.sentiment_collector = SentimentCollector(self.config)
            self.logger.info("Collecteur de sentiment initialisé avec succès")

            # self.macro_collector = MacroCollector(self.base_path / "macro", self.config)
            # self.volume_collector = VolumeCollector(self.base_path / "volume_flow", self.config)
            # self.alternative_collector = AlternativeCollector(self.base_path / "alternative", self.config)
            # self.options_collector = OptionsCollector(self.base_path / "price_data/options", self.config)
            # self.real_time_collector = RealTimeCollector(self.base_path / "real_time", self.config)

        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation des collecteurs: {e}")
            raise

    def collect_all(self):
        self.logger.info("Début de la collecte des données...")

        try:
            self.price_collector.collect()
            self.logger.info("✅ Collecte des données de prix réussie.")
        except Exception as e:
            self.logger.error(f"❌ Erreur collecte des données de prix: {e}")

        try:
            self.fundamental_collector.collect()
            self.logger.info("✅ Collecte des données fondamentales réussie.")
        except Exception as e:
            self.logger.error(f"❌ Erreur collecte des données fondamentales: {e}")

        try:
            self.sentiment_collector.collect()
            self.logger.info("✅ Collecte des données de sentiment réussie.")
        except Exception as e:
            self.logger.error(f"❌ Erreur collecte des données de sentiment: {e}")

    def validate_config(self):
        if not self.config:
            return False

        api_keys = self.config.get('api_keys', {})
        missing_keys = []

        for key, value in api_keys.items():
            if not value or value.startswith('YOUR_'):
                missing_keys.append(key)

        if missing_keys:
            self.logger.warning(f"Clés API manquantes ou non configurées: {missing_keys}")
            self.logger.warning("Veuillez modifier le fichier config/apikey.yaml avec vos vraies clés API")
            return False

        return True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('collector.log', encoding='utf-8')
        ]
    )

    try:
        manager = CollectorManager()

        if not manager.validate_config():
            print("\n⚠️ ATTENTION: Certaines clés API ne sont pas configurées.")
            print("Veuillez modifier le fichier 'config/apikey.yaml' avec vos vraies clés API.\n")

        manager.collect_all()

    except Exception as e:
        logging.error(f"Erreur fatale: {e}")
        print(f"\n❌ Erreur fatale: {e}")
        print("Vérifiez les logs pour plus de détails.")
