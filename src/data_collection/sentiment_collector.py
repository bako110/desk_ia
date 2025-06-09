# =============================================================================
# SENTIMENT COLLECTOR - NASDAQ IA TRADING
# Version restructurée avec gestion automatique des dossiers
# =============================================================================
from pathlib import Path
import yaml
import logging
import time
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np
from textblob import TextBlob
import json
import os
import sys


class SentimentCollector:
    """Collecteur de données de sentiment et d'actualités financières"""

    def __init__(self, config: dict = None, base_data_folder: Path = None):
        """
        Initialise le collecteur de sentiment
        
        Args:
            config: Configuration avec les clés API
            base_data_folder: Dossier de base des données (défaut: ./data)
        """
        # Configuration des dossiers
        self.base_data_folder = Path(base_data_folder) if base_data_folder else Path("data")
        self.raw_data_folder = self.base_data_folder / "raw"
        self.sentiment_folder = self.raw_data_folder / "sentiments"
        
        # Créer la structure de dossiers si elle n'existe pas
        self._create_directory_structure()
        
        # Configuration des APIs
        if config:
            self.sentiment_apis = config.get('sentiment_apis', {})
            self.rate_limits = config.get('rate_limits', {})
        else:
            # Configuration par défaut avec les clés disponibles
            self.sentiment_apis = {
                'alpha_vantage': 'RU6W0PWAUZ0JYD0A',
                'finnhub': 'd0ng2fpr01qi1cve64bgd0ng2fpr01qi1cve64c0'
            }
            self.rate_limits = {
                'alpha_vantage': 5,
                'finnhub': 60
            }
        
        # Configuration du logging
        self._setup_logging()
        
        # Gestion du rate limiting
        self.request_times = {}
        
        # Log des APIs disponibles
        self._log_available_apis()
        
    def _create_directory_structure(self):
        """Créer la structure de dossiers nécessaire"""
        try:
            # Créer les dossiers principaux
            self.base_data_folder.mkdir(parents=True, exist_ok=True)
            self.raw_data_folder.mkdir(parents=True, exist_ok=True)
            self.sentiment_folder.mkdir(parents=True, exist_ok=True)
            
            # Créer des sous-dossiers pour l'organisation
            (self.sentiment_folder / "daily").mkdir(parents=True, exist_ok=True)
            (self.sentiment_folder / "historical").mkdir(parents=True, exist_ok=True)
            (self.sentiment_folder / "composite").mkdir(parents=True, exist_ok=True)
            (self.sentiment_folder / "logs").mkdir(parents=True, exist_ok=True)
            
            print(f"✅ Structure de dossiers créée: {self.sentiment_folder}")
            
        except Exception as e:
            print(f"❌ Erreur création dossiers: {e}")
            raise
    
    def _setup_logging(self):
        """Configuration du système de logging"""
        # Fichier de log avec horodatage
        log_filename = f"sentiment_collector_{datetime.now().strftime('%Y%m%d')}.log"
        log_filepath = self.sentiment_folder / "logs" / log_filename
        
        # Configuration du logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filepath, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialisé - Fichier: {log_filepath}")
    
    def _log_available_apis(self):
        """Logger les APIs disponibles"""
        available_apis = []
        if self.sentiment_apis.get('alpha_vantage'):
            available_apis.append('Alpha Vantage')
        if self.sentiment_apis.get('finnhub'):
            available_apis.append('Finnhub')
        
        self.logger.info(f"SentimentCollector initialisé avec APIs: {', '.join(available_apis)}")
        self.logger.info(f"Dossier de données: {self.sentiment_folder}")

    async def _rate_limit_wait(self, api_name: str):
        """Gestion du rate limiting"""
        current_time = time.time()
        if api_name not in self.request_times:
            self.request_times[api_name] = []

        # Nettoyer les anciennes requêtes (> 1 minute)
        self.request_times[api_name] = [
            req_time for req_time in self.request_times[api_name]
            if current_time - req_time < 60
        ]

        # Vérifier limite
        limit = self.rate_limits.get(api_name, 60)
        if len(self.request_times[api_name]) >= limit:
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
                        data = await response.json()
                        return data
                    elif response.status == 429:
                        self.logger.warning(f"Rate limit dépassé pour {api_name}")
                        return {"error": "Rate limit exceeded"}
                    else:
                        self.logger.error(f"Erreur API {api_name}: Status {response.status}")
                        return {"error": f"HTTP {response.status}"}
            except asyncio.TimeoutError:
                self.logger.error(f"Timeout pour {api_name}")
                return {"error": "Request timeout"}
            except Exception as e:
                self.logger.error(f"Erreur requête {api_name}: {str(e)}")
                return {"error": str(e)}
    
    async def get_alphavantage_news_sentiment(self, symbol: str, limit: int = 50) -> dict:
        """Collecter sentiment des news Alpha Vantage"""
        if not self.sentiment_apis.get('alpha_vantage'):
            return {"error": "Alpha Vantage API key not available", "symbol": symbol}
            
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "limit": limit,
            "apikey": self.sentiment_apis["alpha_vantage"]
        }
        
        self.logger.info(f"Récupération news Alpha Vantage pour {symbol}")
        data = await self._make_request(url, params, "alpha_vantage")
        
        if "error" in data:
            return {"error": data["error"], "symbol": symbol}
        
        # Vérifier les erreurs spécifiques Alpha Vantage
        if "Error Message" in data:
            return {"error": data["Error Message"], "symbol": symbol}
        
        if "Note" in data:
            return {"error": "API call frequency limit reached", "symbol": symbol}
        
        if "feed" in data and isinstance(data["feed"], list):
            # Analyser chaque article
            analyzed_news = []
            for article in data["feed"]:
                try:
                    sentiment_analysis = {
                        "title": article.get("title", ""),
                        "summary": article.get("summary", ""),
                        "url": article.get("url", ""),
                        "time_published": article.get("time_published", ""),
                        "source": article.get("source", ""),
                        "overall_sentiment_score": self._safe_float(article.get("overall_sentiment_score", 0)),
                        "overall_sentiment_label": article.get("overall_sentiment_label", "Neutral"),
                        "ticker_sentiment": article.get("ticker_sentiment", [])
                    }
                    analyzed_news.append(sentiment_analysis)
                except Exception as e:
                    self.logger.warning(f"Erreur analyse article {symbol}: {e}")
                    continue
            
            result = {
                "symbol": symbol,
                "source": "alpha_vantage",
                "news_count": len(analyzed_news),
                "articles": analyzed_news,
                "avg_sentiment_score": np.mean([art["overall_sentiment_score"] for art in analyzed_news]) if analyzed_news else 0,
                "timestamp": datetime.now().isoformat()
            }
            
            # Sauvegarder automatiquement
            await self._save_sentiment_data(result, f"alphavantage_news_{symbol}")
            return result
        
        return {"error": "No valid news data received", "symbol": symbol}
    
    async def get_finnhub_news(self, symbol: str, days_back: int = 7) -> dict:
        """Collecter news Finnhub"""
        if not self.sentiment_apis.get('finnhub'):
            return {"error": "Finnhub API key not available", "symbol": symbol}
            
        today = datetime.now()
        start_date = today - timedelta(days=days_back)
        
        url = "https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": symbol,
            "from": start_date.strftime("%Y-%m-%d"),
            "to": today.strftime("%Y-%m-%d"),
            "token": self.sentiment_apis["finnhub"]
        }
        
        self.logger.info(f"Récupération news Finnhub pour {symbol}")
        data = await self._make_request(url, params, "finnhub")
        
        if "error" in data:
            return {"error": data["error"], "symbol": symbol}
        
        if isinstance(data, list) and len(data) > 0:
            analyzed_news = []
            for article in data[:50]:  # Limiter à 50 articles
                try:
                    # Analyser sentiment avec TextBlob
                    headline = article.get("headline", "")
                    summary = article.get("summary", "")
                    
                    title_sentiment = TextBlob(headline).sentiment if headline else TextBlob("").sentiment
                    summary_sentiment = TextBlob(summary).sentiment if summary else TextBlob("").sentiment
                    
                    news_item = {
                        "headline": headline,
                        "summary": summary,
                        "url": article.get("url", ""),
                        "datetime": article.get("datetime", 0),
                        "source": article.get("source", ""),
                        "title_polarity": title_sentiment.polarity,
                        "title_subjectivity": title_sentiment.subjectivity,
                        "summary_polarity": summary_sentiment.polarity,
                        "summary_subjectivity": summary_sentiment.subjectivity,
                        "image": article.get("image", "")
                    }
                    analyzed_news.append(news_item)
                except Exception as e:
                    self.logger.warning(f"Erreur analyse article Finnhub {symbol}: {e}")
                    continue
            
            if analyzed_news:
                title_polarities = [art["title_polarity"] for art in analyzed_news if art["title_polarity"] is not None]
                summary_polarities = [art["summary_polarity"] for art in analyzed_news if art["summary_polarity"] is not None]
                
                result = {
                    "symbol": symbol,
                    "source": "finnhub",
                    "news_count": len(analyzed_news),
                    "articles": analyzed_news,
                    "avg_title_sentiment": np.mean(title_polarities) if title_polarities else 0,
                    "avg_summary_sentiment": np.mean(summary_polarities) if summary_polarities else 0,
                    "sentiment_std": np.std(title_polarities) if title_polarities else 0,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Sauvegarder automatiquement
                await self._save_sentiment_data(result, f"finnhub_news_{symbol}")
                return result
        
        return {"error": "No valid news data received", "symbol": symbol}
    
    async def get_finnhub_social_sentiment(self, symbol: str) -> dict:
        """Collecter sentiment des réseaux sociaux via Finnhub"""
        if not self.sentiment_apis.get('finnhub'):
            return {"error": "Finnhub API key not available", "symbol": symbol}
            
        url = "https://finnhub.io/api/v1/stock/social-sentiment"
        params = {
            "symbol": symbol,
            "token": self.sentiment_apis["finnhub"]
        }
        
        self.logger.info(f"Récupération sentiment social pour {symbol}")
        data = await self._make_request(url, params, "finnhub")
        
        if "error" in data:
            return {"error": data["error"], "symbol": symbol}
        
        if "reddit" in data or "twitter" in data:
            result = {
                "symbol": symbol,
                "source": "finnhub_social",
                "reddit_sentiment": data.get("reddit", []),
                "twitter_sentiment": data.get("twitter", []),
                "timestamp": datetime.now().isoformat()
            }
            
            # Calculer scores moyens
            try:
                if "reddit" in data and isinstance(data["reddit"], list):
                    reddit_scores = [item.get("score", 0) for item in data["reddit"] if isinstance(item, dict)]
                    result["reddit_avg_score"] = np.mean(reddit_scores) if reddit_scores else 0
                    result["reddit_mentions"] = len(data["reddit"])
                else:
                    result["reddit_avg_score"] = 0
                    result["reddit_mentions"] = 0
                
                if "twitter" in data and isinstance(data["twitter"], list):
                    twitter_scores = [item.get("score", 0) for item in data["twitter"] if isinstance(item, dict)]
                    result["twitter_avg_score"] = np.mean(twitter_scores) if twitter_scores else 0
                    result["twitter_mentions"] = len(data["twitter"])
                else:
                    result["twitter_avg_score"] = 0
                    result["twitter_mentions"] = 0
                    
            except Exception as e:
                self.logger.warning(f"Erreur calcul sentiment social {symbol}: {e}")
                result["reddit_avg_score"] = 0
                result["twitter_avg_score"] = 0
                result["reddit_mentions"] = 0
                result["twitter_mentions"] = 0
            
            # Sauvegarder automatiquement
            await self._save_sentiment_data(result, f"finnhub_social_{symbol}")
            return result
        
        return {"error": "No social sentiment data available", "symbol": symbol}
    
    def calculate_composite_sentiment_score(self, news_av_data: dict, news_fh_data: dict, social_data: dict) -> dict:
        """Calculer un score de sentiment composite"""
        scores = []
        weights = []
        components = {}
        
        # Score des news Alpha Vantage
        if not news_av_data.get("error") and "avg_sentiment_score" in news_av_data:
            av_score = news_av_data["avg_sentiment_score"]
            if av_score != 0:
                scores.append(av_score)
                weights.append(0.4)  # 40% de poids
                components["alphavantage_news"] = av_score
        
        # Score des news Finnhub
        if not news_fh_data.get("error") and "avg_title_sentiment" in news_fh_data:
            fh_score = news_fh_data["avg_title_sentiment"]
            if fh_score != 0:
                scores.append(fh_score)
                weights.append(0.3)  # 30% de poids
                components["finnhub_news"] = fh_score
        
        # Score social Reddit
        if not social_data.get("error") and "reddit_avg_score" in social_data:
            reddit_score = social_data["reddit_avg_score"]
            if reddit_score != 0:
                scores.append(reddit_score)
                weights.append(0.2)  # 20% de poids
                components["reddit_sentiment"] = reddit_score
        
        # Score social Twitter
        if not social_data.get("error") and "twitter_avg_score" in social_data:
            twitter_score = social_data["twitter_avg_score"]
            if twitter_score != 0:
                scores.append(twitter_score)
                weights.append(0.1)  # 10% de poids
                components["twitter_sentiment"] = twitter_score
        
        if scores and weights:
            # Normaliser les poids
            total_weight = sum(weights)
            weights = [w/total_weight for w in weights]
            
            # Score pondéré
            composite_score = np.average(scores, weights=weights)
            
            # Normaliser entre -1 et 1
            composite_score = max(-1, min(1, composite_score))
            
            # Classification
            if composite_score > 0.1:
                sentiment_label = "Bullish"
            elif composite_score < -0.1:
                sentiment_label = "Bearish"
            else:
                sentiment_label = "Neutral"
            
            return {
                "composite_score": float(composite_score),
                "sentiment_label": sentiment_label,
                "confidence": float(abs(composite_score)),
                "components": components,
                "data_sources": len(scores)
            }
        
        return {
            "composite_score": 0.0,
            "sentiment_label": "Neutral",
            "confidence": 0.0,
            "components": {},
            "data_sources": 0,
            "note": "Insufficient data for composite score"
        }
    
    async def get_market_fear_greed_index(self) -> dict:
        """Calculer un index Fear & Greed basé sur VIX et volatilité du marché"""
        try:
            # En production, vous pourriez récupérer le VIX réel
            # Pour l'exemple, on simule avec des données réalistes
            vix_value = np.random.uniform(15, 35)  # VIX typique entre 15-35
            
            # Conversion VIX en Fear & Greed (inversé)
            # VIX élevé = Fear, VIX bas = Greed
            if vix_value > 30:
                fear_greed_score = np.random.randint(0, 25)  # Extreme Fear
                label = "Extreme Fear"
            elif vix_value > 25:
                fear_greed_score = np.random.randint(25, 45)  # Fear
                label = "Fear"
            elif vix_value > 20:
                fear_greed_score = np.random.randint(45, 55)  # Neutral
                label = "Neutral"
            elif vix_value > 15:
                fear_greed_score = np.random.randint(55, 75)  # Greed
                label = "Greed"
            else:
                fear_greed_score = np.random.randint(75, 100)  # Extreme Greed
                label = "Extreme Greed"
            
            result = {
                "fear_greed_score": fear_greed_score,
                "label": label,
                "vix_reference": vix_value,
                "timestamp": datetime.now().isoformat()
            }
            
            # Sauvegarder automatiquement
            await self._save_sentiment_data(result, "market_fear_greed_index")
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur calcul Fear & Greed: {e}")
            return {
                "fear_greed_score": 50,
                "label": "Neutral",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _safe_float(self, value):
        """Convertir en float de manière sécurisée"""
        try:
            if value is None or value == "":
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    async def analyze_sentiment_trends(self, symbol: str, days: int = 30) -> dict:
        """Analyser les tendances de sentiment sur plusieurs jours (simulé)"""
        try:
            trends = []
            
            for i in range(days):
                date = datetime.now() - timedelta(days=i)
                
                # Simulation de données historiques avec une tendance réaliste
                base_sentiment = np.random.uniform(-0.5, 0.5)
                noise = np.random.uniform(-0.2, 0.2)
                
                daily_sentiment = {
                    "date": date.strftime("%Y-%m-%d"),
                    "sentiment_score": base_sentiment + noise,
                    "news_volume": np.random.randint(5, 50),
                    "social_mentions": np.random.randint(10, 200)
                }
                trends.append(daily_sentiment)
            
            # Calculer moyennes mobiles
            sentiment_scores = [t["sentiment_score"] for t in trends]
            
            # Moyennes mobiles avec gestion des erreurs
            ma_7 = []
            ma_14 = []
            
            if len(sentiment_scores) >= 7:
                ma_7 = np.convolve(sentiment_scores, np.ones(7)/7, mode='valid').tolist()
            
            if len(sentiment_scores) >= 14:
                ma_14 = np.convolve(sentiment_scores, np.ones(14)/14, mode='valid').tolist()
            
            # Déterminer la tendance
            current_trend = "Stable"
            if len(sentiment_scores) >= 8:
                recent_avg = np.mean(sentiment_scores[:7])
                older_avg = np.mean(sentiment_scores[7:14]) if len(sentiment_scores) >= 14 else np.mean(sentiment_scores[7:])
                
                if recent_avg > older_avg + 0.1:
                    current_trend = "Improving"
                elif recent_avg < older_avg - 0.1:
                    current_trend = "Declining"
            
            result = {
                "symbol": symbol,
                "source": "trend_analysis",
                "daily_trends": trends,
                "moving_averages": {
                    "ma_7": ma_7,
                    "ma_14": ma_14
                },
                "current_trend": current_trend,
                "volatility": float(np.std(sentiment_scores)),
                "avg_sentiment": float(np.mean(sentiment_scores)),
                "timestamp": datetime.now().isoformat()
            }
            
            # Sauvegarder automatiquement
            await self._save_sentiment_data(result, f"sentiment_trends_{symbol}")
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur analyse tendances {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}
    
    async def _save_sentiment_data(self, data: dict, filename_prefix: str):
        """Sauvegarder les données de sentiment automatiquement"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{filename_prefix}_{timestamp}.json"
            
            # Déterminer le sous-dossier selon le type de données
            if "composite" in filename_prefix.lower():
                subfolder = "composite"
            elif "trend" in filename_prefix.lower():
                subfolder = "historical"
            else:
                subfolder = "daily"
            
            filepath = self.sentiment_folder / subfolder / filename
            
            # Sauvegarder les données
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"✅ Données sauvegardées: {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur sauvegarde données: {e}")
    
    async def collect_comprehensive_sentiment(self, symbols: List[str]) -> Dict[str, dict]:
        """Collecter sentiment complet pour plusieurs symboles avec sauvegarde automatique"""
        results = {}
        
        self.logger.info(f"🚀 Début collecte sentiment pour {len(symbols)} symboles")
        print(f"📁 Dossier de destination: {self.sentiment_folder}")
        
        for symbol in symbols:
            self.logger.info(f"📊 Collecte sentiment pour {symbol}")
            print(f"   ⏳ Traitement de {symbol}...")
            
            try:
                # Collecter toutes les données en parallèle
                tasks = []
                
                # News Alpha Vantage (si disponible)
                if self.sentiment_apis.get('alpha_vantage'):
                    tasks.append(self.get_alphavantage_news_sentiment(symbol))
                else:
                    tasks.append(asyncio.coroutine(lambda: {"error": "Alpha Vantage not available"})())
                
                # News Finnhub (si disponible)
                if self.sentiment_apis.get('finnhub'):
                    tasks.append(self.get_finnhub_news(symbol))
                    tasks.append(self.get_finnhub_social_sentiment(symbol))
                else:
                    tasks.append(asyncio.coroutine(lambda: {"error": "Finnhub not available"})())
                    tasks.append(asyncio.coroutine(lambda: {"error": "Finnhub not available"})())
                
                # Tendances (toujours disponible)
                tasks.append(self.analyze_sentiment_trends(symbol))
                
                results_data = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Traiter les résultats
                news_av = results_data[0] if not isinstance(results_data[0], Exception) else {"error": str(results_data[0])}
                news_fh = results_data[1] if not isinstance(results_data[1], Exception) else {"error": str(results_data[1])}
                social_fh = results_data[2] if not isinstance(results_data[2], Exception) else {"error": str(results_data[2])}
                trends = results_data[3] if not isinstance(results_data[3], Exception) else {"error": str(results_data[3])}
                
                # Calculer score composite
                composite = self.calculate_composite_sentiment_score(news_av, news_fh, social_fh)
                
                # Sauvegarder le score composite
                composite_data = {
                    "symbol": symbol,
                    "composite_sentiment": composite,
                    "timestamp": datetime.now().isoformat()
                }
                await self._save_sentiment_data(composite_data, f"composite_sentiment_{symbol}")
                
                # Index Fear & Greed général (une seule fois par collecte)
                if symbol == symbols[0]:  # Calculer seulement pour le premier symbole
                    fear_greed = await self.get_market_fear_greed_index()
                else:
                    fear_greed = {"note": "Calculated once per collection cycle"}
                
                results[symbol] = {
                    "news_sentiment_alphavantage": news_av,
                    "news_sentiment_finnhub": news_fh,
                    "social_sentiment": social_fh,
                    "sentiment_trends": trends,
                    "composite_sentiment": composite,
                    "market_fear_greed": fear_greed,
                    "collection_timestamp": datetime.now().isoformat()
                }
                
                sentiment_label = composite.get('sentiment_label', 'Unknown')
                confidence = composite.get('confidence', 0)
                self.logger.info(f"✅ Sentiment collecté pour {symbol}: {sentiment_label} (Confiance: {confidence:.2f})")
                print(f"   ✅ {symbol}: {sentiment_label} (Confiance: {confidence:.2f})")
                
            except Exception as e:
                self.logger.error(f"❌ Erreur collecte sentiment {symbol}: {str(e)}")
                print(f"   ❌ Erreur pour {symbol}: {str(e)}")
                results[symbol] = {
                    "error": str(e),
                    "collection_timestamp": datetime.now().isoformat()
                }
        
        # Sauvegarder le rapport complet
        await self._save_comprehensive_report(results)
        
        self.logger.info(f"🎉 Collecte sentiment terminée pour {len(results)} symboles")
        print(f"🎉 Collecte terminée! Fichiers sauvegardés dans: {self.sentiment_folder}")
        
        return results
    
    async def _save_comprehensive_report(self, results: dict):
        """Sauvegarder un rapport complet de toutes les données collectées"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"comprehensive_sentiment_report_{timestamp}.json"
            filepath = self.sentiment_folder / filename
            
            # Préparer le rapport avec métadonnées
            report = {
                "collection_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "symbols_count": len(results),
                    "symbols": list(results.keys()),
                    "data_sources": list(self.sentiment_apis.keys()),
                    "collection_duration": "N/A"  # Peut être calculé si nécessaire
                },
                "sentiment_data": results,
                "summary": self._generate_collection_summary(results)
            }
            
            # Sauvegarder le rapport
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"📋 Rapport complet sauvegardé: {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur sauvegarde rapport complet: {e}")
    
    def _generate_collection_summary(self, results: dict) -> dict:
        """Générer un résumé de la collecte"""
        try:
            summary = {
                "total_symbols": len(results),
                "successful_collections": 0,
                "failed_collections": 0,
                "sentiment_distribution": {"Bullish": 0, "Bearish": 0, "Neutral": 0},
                "average_confidence": 0.0,
                "data_source_success": {
                    "alphavantage": 0,
                    "finnhub_news": 0,
                    "finnhub_social": 0,
                    "trends": 0
                }
            }
            
            confidences = []
            
            for symbol, data in results.items():
                if "error" not in data:
                    summary["successful_collections"] += 1
                    
                    # Analyser le sentiment composite
                    composite = data.get("composite_sentiment", {})
                    if composite:
                        label = composite.get("sentiment_label", "Neutral")
                        if label in summary["sentiment_distribution"]:
                            summary["sentiment_distribution"][label] += 1
                        
                        confidence = composite.get("confidence", 0)
                        if confidence > 0:
                            confidences.append(confidence)
                    
                    # Compter les sources de données réussies
                    if not data.get("news_sentiment_alphavantage", {}).get("error"):
                        summary["data_source_success"]["alphavantage"] += 1
                    
                    if not data.get("news_sentiment_finnhub", {}).get("error"):
                        summary["data_source_success"]["finnhub_news"] += 1
                    
                    if not data.get("social_sentiment", {}).get("error"):
                        summary["data_source_success"]["finnhub_social"] += 1
                    
                    if not data.get("sentiment_trends", {}).get("error"):
                        summary["data_source_success"]["trends"] += 1
                        
                else:
                    summary["failed_collections"] += 1
            
            # Calculer la confiance moyenne
            if confidences:
                summary["average_confidence"] = float(np.mean(confidences))
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Erreur génération résumé: {e}")
            return {"error": str(e)}
    
    def collect(self, symbols: List[str] = None):
        """Méthode principale de collecte synchrone - interface commune"""
        if symbols is None:
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        print(f"🚀 Démarrage de la collecte de sentiment pour {len(symbols)} symboles")
        print(f"📁 Dossier de destination: {self.sentiment_folder}")
        
        # Exécuter la collecte asynchrone
        return asyncio.run(self.collect_comprehensive_sentiment(symbols))
    
    def get_latest_sentiment_data(self, symbol: str = None) -> dict:
        """Récupérer les dernières données de sentiment sauvegardées"""
        try:
            if symbol:
                # Chercher les fichiers pour un symbole spécifique
                pattern = f"*{symbol}*.json"
                files = list(self.sentiment_folder.glob(f"**/{pattern}"))
            else:
                # Chercher tous les fichiers de rapport complet
                files = list(self.sentiment_folder.glob("comprehensive_sentiment_report_*.json"))
            
            if not files:
                return {"error": f"Aucun fichier trouvé pour {symbol if symbol else 'rapport complet'}"}
            
            # Trouver le fichier le plus récent
            latest_file = max(files, key=os.path.getctime)
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return {
                "file_path": str(latest_file),
                "data": data,
                "last_modified": datetime.fromtimestamp(os.path.getctime(latest_file)).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Erreur récupération données: {e}")
            return {"error": str(e)}
    
    def list_available_data(self) -> dict:
        """Lister toutes les données disponibles dans le dossier sentiment"""
        try:
            data_inventory = {
                "daily": [],
                "historical": [],
                "composite": [],
                "reports": []
            }
            
            # Scanner chaque sous-dossier
            for subfolder in ["daily", "historical", "composite"]:
                folder_path = self.sentiment_folder / subfolder
                if folder_path.exists():
                    files = list(folder_path.glob("*.json"))
                    for file in files:
                        file_info = {
                            "filename": file.name,
                            "path": str(file),
                            "size_kb": round(file.stat().st_size / 1024, 2),
                            "created": datetime.fromtimestamp(file.stat().st_ctime).isoformat(),
                            "modified": datetime.fromtimestamp(file.stat().st_mtime).isoformat()
                        }
                        data_inventory[subfolder].append(file_info)
            
            # Scanner les rapports complets
            report_files = list(self.sentiment_folder.glob("comprehensive_sentiment_report_*.json"))
            for file in report_files:
                file_info = {
                    "filename": file.name,
                    "path": str(file),
                    "size_kb": round(file.stat().st_size / 1024, 2),
                    "created": datetime.fromtimestamp(file.stat().st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(file.stat().st_mtime).isoformat()
                }
                data_inventory["reports"].append(file_info)
            
            # Statistiques générales
            total_files = sum(len(files) for files in data_inventory.values())
            total_size = sum(
                sum(file["size_kb"] for file in files) 
                for files in data_inventory.values()
            )
            
            return {
                "inventory": data_inventory,
                "statistics": {
                    "total_files": total_files,
                    "total_size_kb": round(total_size, 2),
                    "folder_structure": str(self.sentiment_folder)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Erreur inventaire données: {e}")
            return {"error": str(e)}
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> dict:
        """Nettoyer les anciennes données (optionnel)"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            cutoff_timestamp = cutoff_date.timestamp()
            
            deleted_files = []
            total_size_freed = 0
            
            # Scanner tous les fichiers JSON
            for json_file in self.sentiment_folder.glob("**/*.json"):
                if json_file.stat().st_mtime < cutoff_timestamp:
                    size_kb = json_file.stat().st_size / 1024
                    deleted_files.append({
                        "filename": json_file.name,
                        "path": str(json_file),
                        "size_kb": round(size_kb, 2)
                    })
                    total_size_freed += size_kb
                    json_file.unlink()  # Supprimer le fichier
            
            self.logger.info(f"🧹 Nettoyage terminé: {len(deleted_files)} fichiers supprimés")
            
            return {
                "files_deleted": len(deleted_files),
                "size_freed_kb": round(total_size_freed, 2),
                "deleted_files": deleted_files,
                "cutoff_date": cutoff_date.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Erreur nettoyage données: {e}")
            return {"error": str(e)}


# =============================================================================
# FONCTION PRINCIPALE POUR TESTER LA COLLECTE
# =============================================================================
async def main():
    """Fonction principale pour tester le collecteur"""
    print("=" * 60)
    print("🚀 SENTIMENT COLLECTOR - NASDAQ IA TRADING")
    print("=" * 60)
    
    # Initialiser le collecteur
    collector = SentimentCollector()
    
    # Symboles à analyser
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    try:
        # Collecter les données
        results = await collector.collect_comprehensive_sentiment(symbols)
        
        # Afficher un résumé
        print(f"\n📊 RÉSUMÉ DE LA COLLECTE:")
        print(f"   • Symboles traités: {len(results)}")
        
        for symbol, data in results.items():
            if "error" not in data:
                composite = data.get("composite_sentiment", {})
                sentiment = composite.get("sentiment_label", "Unknown")
                confidence = composite.get("confidence", 0)
                print(f"   • {symbol}: {sentiment} (Confiance: {confidence:.2f})")
            else:
                print(f"   • {symbol}: ERREUR - {data.get('error', 'Unknown')}")
        
        # Afficher l'inventaire des fichiers
        print(f"\n📁 FICHIERS CRÉÉS:")
        inventory = collector.list_available_data()
        stats = inventory.get("statistics", {})
        print(f"   • Total fichiers: {stats.get('total_files', 0)}")
        print(f"   • Taille totale: {stats.get('total_size_kb', 0)} KB")
        print(f"   • Dossier: {collector.sentiment_folder}")
        
        return results
        
    except Exception as e:
        print(f"❌ Erreur lors de la collecte: {e}")
        return {}


# =============================================================================
# FONCTION SYNCHRONE POUR COMPATIBILITÉ
# =============================================================================
def run_sentiment_collection(symbols: List[str] = None):
    """Fonction synchrone pour lancer la collecte"""
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    collector = SentimentCollector()
    return collector.collect(symbols)


# =============================================================================
# POINT D'ENTRÉE POUR EXÉCUTION DIRECTE
# =============================================================================
if __name__ == "__main__":
    # Exécuter la collecte de test
    asyncio.run(main())