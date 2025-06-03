import logging
from pathlib import Path

from src.data_collection.price_collector import PriceCollector
from src.data_collection.fundamental_collector import FundamentalCollector
from src.data_collection.sentiment_collector import SentimentCollector
from src.data_collection.macro_collector import MacroCollector
from src.data_collection.volume_collector import VolumeCollector
from src.data_collection.alternative_collector import AlternativeCollector
from src.data_collection.options_collector import OptionsCollector
from src.data_collection.real_time_collector import RealTimeCollector


class CollectorManager:
    def __init__(self, base_data_dir: str = "data/raw"):
        self.base_path = Path(base_data_dir)
        self.logger = logging.getLogger(__name__)
        self._init_collectors()

    def _init_collectors(self):
        self.price_collector = PriceCollector(self.base_path / "price_data")
        self.fundamental_collector = FundamentalCollector(self.base_path / "fundamental")
        self.sentiment_collector = SentimentCollector(self.base_path / "sentiment")
        self.macro_collector = MacroDataCollector(self.base_path / "macro")
        self.volume_collector = VolumeCollector(self.base_path / "volume_flow")
        self.alternative_collector = AlternativeCollector(self.base_path / "alternative")
        self.options_collector = OptionsCollector(self.base_path / "price_data/options")
        self.real_time_collector = RealTimeCollector(self.base_path / "real_time")

    def collect_all(self):
        self.logger.info("Début de la collecte de toutes les données...")

        try:
            self.price_collector.collect()
            self.logger.info("Collecte des données prix terminée.")
        except Exception as e:
            self.logger.error(f"Erreur collecte prix: {e}")

        try:
            self.fundamental_collector.collect()
            self.logger.info("Collecte des données fondamentales terminée.")
        except Exception as e:
            self.logger.error(f"Erreur collecte fondamentaux: {e}")

        try:
            self.sentiment_collector.collect()
            self.logger.info("Collecte des données sentiment terminée.")
        except Exception as e:
            self.logger.error(f"Erreur collecte sentiment: {e}")

        try:
            self.macro_collector.collect()
            self.logger.info("Collecte des données macro terminée.")
        except Exception as e:
            self.logger.error(f"Erreur collecte macro: {e}")

        try:
            self.volume_collector.collect()
            self.logger.info("Collecte des données volume terminée.")
        except Exception as e:
            self.logger.error(f"Erreur collecte volume: {e}")

        try:
            self.alternative_collector.collect()
            self.logger.info("Collecte des données alternatives terminée.")
        except Exception as e:
            self.logger.error(f"Erreur collecte alternative: {e}")

        try:
            self.options_collector.collect()
            self.logger.info("Collecte des données options terminée.")
        except Exception as e:
            self.logger.error(f"Erreur collecte options: {e}")

        try:
            self.real_time_collector.collect()
            self.logger.info("Collecte des données temps réel terminée.")
        except Exception as e:
            self.logger.error(f"Erreur collecte temps réel: {e}")

        self.logger.info("Collecte complète terminée.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    manager = CollectorManager()
    manager.collect_all()
