# src/data_collection/real_time_collector.py

import requests
import json
import logging

class RealTimeCollector:
    def __init__(self, api_url, api_key=None):
        self.api_url = api_url
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)

    def fetch_data(self, params=None):
        """
        Récupère les données en temps réel depuis une API.

        Args:
            params (dict): Paramètres pour la requête API.

        Returns:
            dict: Données JSON récupérées.
        """
        headers = {}
        if self.api_key:
            headers['Authorization'] = f"Bearer {self.api_key}"

        try:
            response = requests.get(self.api_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            self.logger.info(f"Données récupérées avec succès depuis {self.api_url}")
            return data
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Erreur lors de la récupération des données : {e}")
            return None

    def process_data(self, data):
        """
        Traite les données récupérées.

        Args:
            data (dict): Données à traiter.

        Returns:
            dict: Données traitées.
        """
        if not data:
            self.logger.warning("Aucune donnée à traiter.")
            return None

        # Exemple simple de traitement (à adapter selon les besoins)
        processed = {
            "timestamp": data.get("timestamp"),
            "value": data.get("value"),
        }
        self.logger.info("Données traitées.")
        return processed
