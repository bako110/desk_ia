# =============================================================================
# SENTIMENT COLLECTOR - NASDAQ IA TRADING
# =============================================================================
from pathlib import Path
import yaml
import logging
import time
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict
import numpy as np
from textblob import TextBlob
import json



class SentimentCollector:
    """Collecteur de données de sentiment et d'actualités financières"""

    def __init__(self, data_folder: Path):
        self.data_folder = data_folder
        self.config_path = self.data_folder / "api_keys.yaml"

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        self.sentiment_apis = config.get('sentiment_apis', {})
        self.rate_limits = config.get('rate_limits', {})
        self.logger = logging.getLogger(__name__)
        self.request_times = {}
        
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
        if len(self.request_times[api_name]) >= self.rate_limits.get(api_name, 60):
            sleep_time = 60 - (current_time - self.request_times[api_name][0])
            if sleep_time > 0:
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
                        return {}
            except Exception as e:
                self.logger.error(f"Erreur requête {api_name}: {str(e)}")
                return {}
    
    async def get_alphavantage_news_sentiment(self, symbol: str, limit: int = 50) -> dict:
        """Collecter sentiment des news Alpha Vantage"""
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "limit": limit,
            "apikey": self.sentiment_apis["alphavantage"]
        }
        
        data = await self._make_request(url, params, "alphavantage")
        
        if "feed" in data:
            # Analyser chaque article
            analyzed_news = []
            for article in data["feed"]:
                sentiment_analysis = {
                    "title": article.get("title", ""),
                    "summary": article.get("summary", ""),
                    "url": article.get("url", ""),
                    "time_published": article.get("time_published", ""),
                    "source": article.get("source", ""),
                    "overall_sentiment_score": float(article.get("overall_sentiment_score", 0)),
                    "overall_sentiment_label": article.get("overall_sentiment_label", "Neutral"),
                    "ticker_sentiment": article.get("ticker_sentiment", [])
                }
                analyzed_news.append(sentiment_analysis)
            
            return {
                "symbol": symbol,
                "news_count": len(analyzed_news),
                "articles": analyzed_news,
                "timestamp": datetime.now().isoformat()
            }
        
        return {"error": "No data received", "symbol": symbol}
    
    async def get_finnhub_news(self, symbol: str, days_back: int = 7) -> dict:
        """Collecter news Finnhub"""
        today = datetime.now()
        start_date = today - timedelta(days=days_back)
        
        url = "https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": symbol,
            "from": start_date.strftime("%Y-%m-%d"),
            "to": today.strftime("%Y-%m-%d"),
            "token": self.sentiment_apis["finnhub"]
        }
        
        data = await self._make_request(url, params, "finnhub")
        
        if isinstance(data, list):
            analyzed_news = []
            for article in data[:50]:  # Limiter à 50 articles
                # Analyser sentiment avec TextBlob
                title_sentiment = TextBlob(article.get("headline", "")).sentiment
                summary_sentiment = TextBlob(article.get("summary", "")).sentiment
                
                news_item = {
                    "headline": article.get("headline", ""),
                    "summary": article.get("summary", ""),
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
            
            return {
                "symbol": symbol,
                "news_count": len(analyzed_news),
                "articles": analyzed_news,
                "avg_title_sentiment": np.mean([art["title_polarity"] for art in analyzed_news]),
                "avg_summary_sentiment": np.mean([art["summary_polarity"] for art in analyzed_news]),
                "timestamp": datetime.now().isoformat()
            }
        
        return {"error": "No data received", "symbol": symbol}
    
    async def get_finnhub_social_sentiment(self, symbol: str) -> dict:
        """Collecter sentiment des réseaux sociaux"""
        url = "https://finnhub.io/api/v1/stock/social-sentiment"
        params = {
            "symbol": symbol,
            "token": self.sentiment_apis["finnhub"]
        }
        
        data = await self._make_request(url, params, "finnhub")
        
        if "reddit" in data or "twitter" in data:
            result = {
                "symbol": symbol,
                "reddit_sentiment": data.get("reddit", {}),
                "twitter_sentiment": data.get("twitter", {}),
                "timestamp": datetime.now().isoformat()
            }
            
            # Calculer scores moyens
            if "reddit" in data:
                reddit_data = data["reddit"]
                result["reddit_avg_score"] = np.mean([item.get("score", 0) for item in reddit_data])
                result["reddit_mentions"] = len(reddit_data)
            
            if "twitter" in data:
                twitter_data = data["twitter"]
                result["twitter_avg_score"] = np.mean([item.get("score", 0) for item in twitter_data])
                result["twitter_mentions"] = len(twitter_data)
            
            return result
        
        return {"error": "No social sentiment data", "symbol": symbol}
    
    def calculate_composite_sentiment_score(self, news_data: dict, social_data: dict) -> dict:
        """Calculer un score de sentiment composite"""
        scores = []
        weights = []
        
        # Score des news Alpha Vantage
        if "articles" in news_data:
            av_scores = [art.get("overall_sentiment_score", 0) for art in news_data["articles"]]
            if av_scores:
                scores.append(np.mean(av_scores))
                weights.append(0.4)  # 40% de poids
        
        # Score des news Finnhub
        if "avg_title_sentiment" in news_data:
            scores.append(news_data["avg_title_sentiment"])
            weights.append(0.3)  # 30% de poids
        
        # Score social Reddit
        if "reddit_avg_score" in social_data:
            scores.append(social_data["reddit_avg_score"])
            weights.append(0.2)  # 20% de poids
        
        # Score social Twitter
        if "twitter_avg_score" in social_data:
            scores.append(social_data["twitter_avg_score"])
            weights.append(0.1)  # 10% de poids
        
        if scores and weights:
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
                "composite_score": composite_score,
                "sentiment_label": sentiment_label,
                "confidence": abs(composite_score),
                "components": {
                    "scores": scores,
                    "weights": weights
                }
            }
        
        return {
            "composite_score": 0.0,
            "sentiment_label": "Neutral",
            "confidence": 0.0,
            "error": "Insufficient data for composite score"
        }
    
    async def get_market_fear_greed_index(self) -> dict:
        """Simuler un index Fear & Greed (placeholder)"""
        # En réalité, vous pourriez utiliser CNN Fear & Greed Index API
        # ou créer votre propre calcul basé sur VIX, volumes, etc.
        
        fear_greed_score = np.random.randint(0, 100)
        
        if fear_greed_score < 25:
            label = "Extreme Fear"
        elif fear_greed_score < 45:
            label = "Fear"
        elif fear_greed_score < 55:
            label = "Neutral"
        elif fear_greed_score < 75:
            label = "Greed"
        else:
            label = "Extreme Greed"
        
        return {
            "fear_greed_score": fear_greed_score,
            "label": label,
            "timestamp": datetime.now().isoformat()
        }
    
    async def analyze_sentiment_trends(self, symbol: str, days: int = 30) -> dict:
        """Analyser les tendances de sentiment sur plusieurs jours"""
        trends = []
        
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            
            # Collecter données historiques (simulé)
            daily_sentiment = {
                "date": date.strftime("%Y-%m-%d"),
                "sentiment_score": np.random.uniform(-1, 1),
                "news_volume": np.random.randint(5, 50),
                "social_mentions": np.random.randint(10, 200)
            }
            trends.append(daily_sentiment)
        
        # Calculer moyennes mobiles
        sentiment_scores = [t["sentiment_score"] for t in trends]
        ma_7 = np.convolve(sentiment_scores, np.ones(7)/7, mode='valid')
        ma_14 = np.convolve(sentiment_scores, np.ones(14)/14, mode='valid')
        
        return {
            "symbol": symbol,
            "daily_trends": trends,
            "moving_averages": {
                "ma_7": ma_7.tolist(),
                "ma_14": ma_14.tolist()
            },
            "current_trend": "Improving" if sentiment_scores[0] > sentiment_scores[7] else "Declining",
            "volatility": np.std(sentiment_scores),
            "timestamp": datetime.now().isoformat()
        }
    
    async def collect_comprehensive_sentiment(self, symbols: List[str]) -> Dict[str, dict]:
        """Collecter sentiment complet pour plusieurs symboles"""
        results = {}
        
        for symbol in symbols:
            self.logger.info(f"Collecte sentiment pour {symbol}")
            
            try:
                # Collecter toutes les données en parallèle
                tasks = [
                    self.get_alphavantage_news_sentiment(symbol),
                    self.get_finnhub_news(symbol),
                    self.get_finnhub_social_sentiment(symbol),
                    self.analyze_sentiment_trends(symbol)
                ]
                
                news_av, news_fh, social_fh, trends = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Gérer les exceptions
                if isinstance(news_av, Exception):
                    news_av = {"error": str(news_av)}
                if isinstance(news_fh, Exception):
                    news_fh = {"error": str(news_fh)}
                if isinstance(social_fh, Exception):
                    social_fh = {"error": str(social_fh)}
                if isinstance(trends, Exception):
                    trends = {"error": str(trends)}
                
                # Calculer score composite
                composite = self.calculate_composite_sentiment_score(news_av, social_fh)
                
                # Index Fear & Greed général
                fear_greed = await self.get_market_fear_greed_index()
                
                results[symbol] = {
                    "news_sentiment_alphavantage": news_av,
                    "news_sentiment_finnhub": news_fh,
                    "social_sentiment": social_fh,
                    "sentiment_trends": trends,
                    "composite_sentiment": composite,
                    "market_fear_greed": fear_greed,
                    "collection_timestamp": datetime.now().isoformat()
                }
                
                self.logger.info(f"Sentiment collecté pour {symbol}: {composite.get('sentiment_label', 'Unknown')}")
                
            except Exception as e:
                self.logger.error(f"Erreur collecte sentiment {symbol}: {str(e)}")
                results[symbol] = {"error": str(e)}
        
        return results
    
    def export_sentiment_data(self, data: dict, filename: str = None) -> str:
        """Exporter les données de sentiment"""
        if filename is None:
            filename = f"sentiment_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        return filename

# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

async def main():
    """Exemple d'utilisation du collecteur de sentiment"""
    collector = SentimentCollector()
    
    # Symboles à analyser
    symbols = ["AAPL", "GOOGL", "TSLA", "NVDA"]
    
    print("Début de l'analyse de sentiment...")
    sentiment_data = await collector.collect_comprehensive_sentiment(symbols)
    
    # Afficher résultats
    for symbol, data in sentiment_data.items():
        if "composite_sentiment" in data:
            composite = data["composite_sentiment"]
            print(f"{symbol}: {composite['sentiment_label']} (Score: {composite['composite_score']:.3f})")
        else:
            print(f"{symbol}: Erreur dans la collecte")
    
    # Exporter les données
    filename = collector.export_sentiment_data(sentiment_data)
    print(f"Données exportées vers: {filename}")
