# =============================================================================
# MACRO DATA COLLECTOR - RÉCUPÉRATION DE DONNÉES UNIQUEMENT
# =============================================================================

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Any, Union
import yaml
import asyncio
import aiohttp
import json
from pathlib import Path

class MacroCollector:
    """Collecteur de données macro-économiques - Récupération uniquement"""
    
    def __init__(self, config: Union[dict, str, Path] = None, data_folder: Path = None):
        """
        Initialise le collecteur de données macro-économiques
        
        Args:
            config: Configuration avec les clés API (dict) ou chemin vers fichier config (str/Path)
            data_folder: Dossier de données (optionnel, pour compatibilité)
        """
        # Configuration du logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialiser les variables
        self.macro_apis = {}
        self.rate_limits = {}
        
        # Traiter la configuration
        if config is None:
            # Essayer de charger depuis le fichier de configuration par défaut
            base_dir = Path(__file__).resolve().parent.parent.parent  # remonte jusqu'à 'desk_ia'
            config_file = base_dir / "config" / "apikeys.yaml"
            self._load_config_from_file(config_file)
        elif isinstance(config, dict):
            # Utiliser la configuration fournie directement
            self.macro_apis = config.get('macro_apis', {})
            self.rate_limits = config.get('rate_limits', {})
        elif isinstance(config, (str, Path)):
            # Charger depuis le fichier spécifié
            config_file = Path(config)
            self._load_config_from_file(config_file)
        else:
            self.logger.error(f"Type de config non supporté: {type(config)}")
            self.macro_apis = {}
            self.rate_limits = {}

        self.data_folder = data_folder or Path("data")
        self.request_times = {}

        # Log des APIs disponibles
        available_apis = []
        if self.macro_apis.get('fred'):
            available_apis.append('FRED')
        
        self.logger.info(f"MacroCollector initialisé avec APIs: {', '.join(available_apis) if available_apis else 'Aucune'}")

        # Indicateurs macro principaux FRED
        self.fred_indicators = {
            "gdp": "GDP",
            "gdp_growth": "A191RL1Q225SBEA",
            "gdp_per_capita": "A939RX0Q048SBEA",
            "real_gdp": "GDPC1",
            "inflation_cpi": "CPIAUCSL",
            "inflation_pce": "PCEPI",
            "core_cpi": "CPILFESL",
            "core_pce": "PCEPILFE",
            "unemployment_rate": "UNRATE",
            "labor_force_participation": "CIVPART",
            "employment_population_ratio": "EMRATIO",
            "nonfarm_payrolls": "PAYEMS",
            "average_hourly_earnings": "AHETPI",
            "federal_funds_rate": "FEDFUNDS",
            "10y_treasury": "GS10",
            "2y_treasury": "GS2",
            "3m_treasury": "GS3M",
            "5y_treasury": "GS5",
            "30y_treasury": "GS30",
            "yield_curve_10y2y": "T10Y2Y",
            "yield_curve_10y3m": "T10Y3M",
            "vix": "VIXCLS",
            "dollar_index": "DTWEXBGS",
            "oil_price": "DCOILWTICO",
            "gold_price": "GOLDAMGBD228NLBM",
            "copper_price": "PCOPPUSDM",
            "consumer_confidence": "UMCSENT",
            "consumer_confidence_present": "UMCSENT1",
            "consumer_confidence_expectations": "UMCSENT5",
            "industrial_production": "INDPRO",
            "capacity_utilization": "TCU",
            "housing_starts": "HOUST",
            "building_permits": "PERMIT",
            "existing_home_sales": "EXHOSLUSM495S",
            "new_home_sales": "HSN1F",
            "retail_sales": "RSXFS",
            "retail_sales_ex_auto": "RSAFS",
            "pmi_manufacturing": "MANEMP",
            "pmi_services": "SRVPRD",
            "initial_claims": "ICSA",
            "continuing_claims": "CCSA",
            "money_supply_m1": "M1SL",
            "money_supply_m2": "M2SL",
            "total_debt": "GFDEBTN",
            "debt_to_gdp": "GFDEGDQ188S",
            "bank_credit": "TOTBKCR",
            "commercial_paper": "COMPAPER",
            "ted_spread": "TEDRATE",
            "credit_spread": "BAMLC0A0CM",
            "mortgage_30y": "MORTGAGE30US",
            "mortgage_15y": "MORTGAGE15US"
        }

    def _load_config_from_file(self, config_file: Path):
        """Charger la configuration depuis un fichier"""
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f)
                
                if isinstance(file_config, dict):
                    self.macro_apis = file_config.get('macro_apis', {})
                    self.rate_limits = file_config.get('rate_limits', {})
                    self.logger.info(f"Configuration chargée depuis: {config_file}")
                else:
                    self.logger.warning(f"Format de config invalide dans {config_file}")
                    self.macro_apis = {}
                    self.rate_limits = {}
            except Exception as e:
                self.logger.warning(f"Erreur lecture config {config_file}: {e}")
                self.macro_apis = {}
                self.rate_limits = {}
        else:
            self.logger.warning(f"Fichier de config non trouvé: {config_file}")
            self.macro_apis = {}
            self.rate_limits = {}

    async def _rate_limit_wait(self, api_name: str):
        """Gestion du rate limiting"""
        current_time = time.time()
        if api_name not in self.request_times:
            self.request_times[api_name] = []
        
        # Nettoyer les anciennes requêtes
        self.request_times[api_name] = [
            req_time for req_time in self.request_times[api_name]
            if current_time - req_time < 60
        ]
        
        # Vérifier limite
        if len(self.request_times[api_name]) >= self.rate_limits.get(api_name, 120):
            sleep_time = 60 - (current_time - self.request_times[api_name][0])
            if sleep_time > 0:
                self.logger.info(f"Rate limit atteint pour {api_name}, attente de {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
        
        self.request_times[api_name].append(current_time)
    
    async def _make_request(self, url: str, params: dict, api_name: str) -> dict:
        """Faire une requête HTTP avec gestion d'erreurs"""
        await self._rate_limit_wait(api_name)
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        self.logger.error(f"Erreur API {api_name}: Status {response.status}")
                        return {"error": f"HTTP {response.status}"}
            except asyncio.TimeoutError:
                self.logger.error(f"Timeout pour {api_name}")
                return {"error": "Request timeout"}
            except Exception as e:
                self.logger.error(f"Erreur requête {api_name}: {str(e)}")
                return {"error": str(e)}
    
    async def get_fred_data(self, series_id: str, limit: int = 1000, 
                           start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Obtenir données FRED (Federal Reserve Economic Data)"""
        if not self.macro_apis.get('fred'):
            self.logger.error("Clé API FRED non disponible")
            return pd.DataFrame()
            
        url = "https://api.stlouisfed.org/fred/series/observations"
        
        params = {
            "series_id": series_id,
            "api_key": self.macro_apis["fred"],
            "file_type": "json",
            "limit": limit
        }
        
        if start_date:
            params["observation_start"] = start_date
        if end_date:
            params["observation_end"] = end_date
        
        data = await self._make_request(url, params, "fred")
        
        if "error" in data:
            self.logger.error(f"Erreur FRED pour {series_id}: {data['error']}")
            return pd.DataFrame()
        
        if "observations" in data:
            df = pd.DataFrame(data["observations"])
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                df = df[["value"]].dropna()
                df.rename(columns={"value": series_id}, inplace=True)
                return df
        
        return pd.DataFrame()
    
    async def get_fred_series_info(self, series_id: str) -> dict:
        """Obtenir informations sur une série FRED"""
        if not self.macro_apis.get('fred'):
            return {"error": "Clé API FRED non disponible"}
            
        url = "https://api.stlouisfed.org/fred/series"
        params = {
            "series_id": series_id,
            "api_key": self.macro_apis["fred"],
            "file_type": "json"
        }
        
        data = await self._make_request(url, params, "fred")
        
        if "error" in data:
            return data
        
        if "seriess" in data and data["seriess"]:
            return data["seriess"][0]
        return {"error": "Aucune information trouvée"}
    
    async def get_single_indicator(self, indicator_name: str, lookback_years: int = 5) -> pd.DataFrame:
        """Récupérer un seul indicateur par son nom"""
        if indicator_name not in self.fred_indicators:
            self.logger.error(f"Indicateur {indicator_name} non trouvé")
            return pd.DataFrame()
        
        series_id = self.fred_indicators[indicator_name]
        start_date = (datetime.now() - timedelta(days=365 * lookback_years)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        self.logger.info(f"Récupération {indicator_name} ({series_id})")
        data = await self.get_fred_data(series_id, limit=2000, 
                                      start_date=start_date, end_date=end_date)
        
        if not data.empty:
            self.logger.info(f"{indicator_name}: {len(data)} observations récupérées")
        else:
            self.logger.warning(f"Aucune donnée pour {indicator_name}")
        
        return data
    
    async def get_multiple_indicators(self, indicator_names: List[str], 
                                    lookback_years: int = 5) -> Dict[str, pd.DataFrame]:
        """Récupérer plusieurs indicateurs"""
        results = {}
        
        for name in indicator_names:
            if name in self.fred_indicators:
                data = await self.get_single_indicator(name, lookback_years)
                if not data.empty:
                    results[name] = data
            else:
                self.logger.warning(f"Indicateur {name} non reconnu")
        
        return results
    
    async def get_all_indicators(self, lookback_years: int = 5) -> Dict[str, pd.DataFrame]:
        """Récupérer tous les indicateurs disponibles"""
        start_date = (datetime.now() - timedelta(days=365 * lookback_years)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        indicators_data = {}
        
        self.logger.info(f"Début de la collecte de {len(self.fred_indicators)} indicateurs")
        
        for name, series_id in self.fred_indicators.items():
            try:
                self.logger.info(f"Collecte {name} ({series_id})")
                data = await self.get_fred_data(series_id, limit=2000, 
                                              start_date=start_date, end_date=end_date)
                
                if not data.empty:
                    indicators_data[name] = data
                    self.logger.info(f"{name}: {len(data)} observations collectées")
                else:
                    self.logger.warning(f"Aucune donnée pour {name}")
                    
            except Exception as e:
                self.logger.error(f"Erreur collecte {name}: {str(e)}")
        
        self.logger.info(f"Collecte terminée: {len(indicators_data)} indicateurs récupérés")
        return indicators_data
    
    def save_data_to_csv(self, data: Dict[str, pd.DataFrame], output_dir: str = "data"):
        """Sauvegarder les données en CSV"""
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for name, df in data.items():
            if not df.empty:
                filename = f"{output_dir}/{name}_{datetime.now().strftime('%Y%m%d')}.csv"
                df.to_csv(filename)
                self.logger.info(f"Sauvegardé: {filename}")
    
    def save_data_to_excel(self, data: Dict[str, pd.DataFrame], filename: str = None):
        """Sauvegarder toutes les données dans un fichier Excel"""
        if filename is None:
            filename = f"macro_data_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            for name, df in data.items():
                if not df.empty:
                    # Limiter le nom de l'onglet à 31 caractères (limite Excel)
                    sheet_name = name[:31] if len(name) > 31 else name
                    df.to_excel(writer, sheet_name=sheet_name)
        
        self.logger.info(f"Données sauvegardées dans: {filename}")
    
    def get_data_summary(self, data: Dict[str, pd.DataFrame]) -> dict:
        """Obtenir un résumé des données collectées"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_indicators": len(data),
            "indicators_summary": {}
        }
        
        for name, df in data.items():
            if not df.empty:
                summary["indicators_summary"][name] = {
                    "observations": len(df),
                    "start_date": df.index.min().isoformat(),
                    "end_date": df.index.max().isoformat(),
                    "latest_value": float(df.iloc[-1].values[0]) if len(df) > 0 else None,
                    "data_quality": "Complete" if len(df) > 50 else "Limited"
                }
            else:
                summary["indicators_summary"][name] = {
                    "observations": 0,
                    "data_quality": "No Data"
                }
        
        return summary
    
    def list_available_indicators(self) -> List[str]:
        """Lister tous les indicateurs disponibles"""
        return list(self.fred_indicators.keys())
    
    def get_indicator_info(self, indicator_name: str) -> dict:
        """Obtenir des informations sur un indicateur spécifique"""
        if indicator_name not in self.fred_indicators:
            return {"error": f"Indicateur {indicator_name} non trouvé"}
        
        return {
            "name": indicator_name,
            "fred_series_id": self.fred_indicators[indicator_name],
            "description": f"Série FRED: {self.fred_indicators[indicator_name]}"
        }

    def collect(self, indicators: List[str] = None, lookback_years: int = 5):
        """Méthode principale de collecte synchrone - interface commune"""
        if indicators is None:
            # Collecter les indicateurs principaux par défaut
            indicators = [
                "gdp_growth", "unemployment_rate", "inflation_cpi", 
                "federal_funds_rate", "10y_treasury", "vix"
            ]
        
        # Exécuter la collecte asynchrone
        return asyncio.run(self.get_multiple_indicators(indicators, lookback_years))

    def export_macro_data(self, data: dict, filename: str = None) -> str:
        """Exporter les données macro-économiques"""
        if filename is None:
            filename = f"macro_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Créer le dossier si nécessaire
        filepath = self.data_folder / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Convertir les DataFrames en format sérialisable
            serializable_data = {}
            for key, df in data.items():
                if isinstance(df, pd.DataFrame):
                    serializable_data[key] = {
                        "data": df.to_dict('records'),
                        "index": df.index.strftime('%Y-%m-%d').tolist() if hasattr(df.index, 'strftime') else df.index.tolist(),
                        "columns": df.columns.tolist()
                    }
                else:
                    serializable_data[key] = df
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Données macro exportées vers: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Erreur export données macro: {e}")
            return ""

# Exemple d'utilisation
async def main():
    """Fonction principale d'exemple"""
    # Créer le collecteur
    collector = MacroCollector()
    
    # Lister les indicateurs disponibles
    print("Indicateurs disponibles:")
    indicators = collector.list_available_indicators()
    for i, indicator in enumerate(indicators, 1):
        print(f"{i:2d}. {indicator}")
    
    print("\n" + "="*50)
    
    # Exemple 1: Récupérer quelques indicateurs spécifiques
    selected_indicators = ["gdp_growth", "unemployment_rate", "inflation_cpi", "federal_funds_rate", "10y_treasury"]
    print(f"Récupération de {len(selected_indicators)} indicateurs...")
    
    data = await collector.get_multiple_indicators(selected_indicators, lookback_years=3)
    
    # Afficher un résumé
    summary = collector.get_data_summary(data)
    print(json.dumps(summary, indent=2, default=str))
    
    # Sauvegarder en Excel
    collector.save_data_to_excel(data, "macro_data_sample.xlsx")
    
    print("\nDonnées sauvegardées avec succès!")

if __name__ == "__main__":
    asyncio.run(main())