#!/usr/bin/env python3
"""
NASDAQ IA Trading - Alternative Data Collector
Collecte les données alternatives: sentiment, news, données satellites, etc.
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
import finnhub
import feedparser
import re
from textblob import TextBlob
import tweepy
from bs4 import BeautifulSoup
import yaml  # Ajouté pour charger le YAML

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/alternative_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AlternativeCollector')

class AlternativeCollector:
    """Collecteur spécialisé pour les données alternatives"""
    
    def __init__(self):
        self.data_path = Path(os.environ.get('DATA_PATH', '/app/data/alternative'))
        self.symbols = os.environ.get('SYMBOLS', 'AAPL,GOOGL,MSFT,TSLA,NVDA').split(',')

        # Charger les clés API depuis le fichier YAML
        self.api_keys = self._load_api_keys('config.api_key.yaml')
        
        # Rate limits
        self.rate_limits = {
            'alphavantage': int(os.environ.get('ALPHAVANTAGE_RATE_LIMIT', 5)),
            'finnhub': int(os.environ.get('FINNHUB_RATE_LIMIT', 60)),
            'social_apis': 100,
            'news_scraping': 50
        }
        
        self._setup_directories()
        self._initialize_clients()
    
    def _load_api_keys(self, yaml_path):
        try:
            with open(yaml_path, 'r') as f:
                keys = yaml.safe_load(f)
            logger.info(f"🔑 Clés API chargées depuis {yaml_path}")
            return keys
        except Exception as e:
            logger.error(f"❌ Erreur chargement clés API depuis {yaml_path}: {e}")
            return {}

    def _setup_directories(self):
        """Crée la structure des répertoires pour les données alternatives"""
        directories = [
            'sentiment_analysis',
            'news_data',
            'social_media',
            'analyst_ratings',
            'earnings_estimates',
            'satellite_data',
            'web_scraping',
            'insider_trading',
            'sec_filings',
            'economic_indicators',
            'crypto_correlation',
            'options_flow',
            'etf_flows',
            'institutional_holdings',
            'alternative_metrics'
        ]
        
        for directory in directories:
            (self.data_path / directory).mkdir(parents=True, exist_ok=True)
            
        logger.info(f"📁 Structure alternative data créée dans {self.data_path}")
    
    def _initialize_clients(self):
        """Initialise les clients API"""
        try:
            # Finnhub client pour news et sentiment
            if self.api_keys.get('finnhub'):
                self.finnhub_client = finnhub.Client(api_key=self.api_keys['finnhub'])
            else:
                self.finnhub_client = None
            
            logger.info("🔌 Clients API alternatifs initialisés")
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation clients: {e}")
    
    def collect_news_sentiment(self):
        """Collecte les news et analyse du sentiment"""
        logger.info("📰 Collecte des news et sentiment...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for symbol in self.symbols:
            try:
                # Collecte des news via Finnhub
                news_data = self._get_finnhub_news(symbol)
                
                # Analyse de sentiment
                sentiment_analysis = self._analyze_news_sentiment(news_data)
                
                # Collecte des news Reddit/autres sources
                reddit_sentiment = self._scrape_reddit_sentiment(symbol)
                
                consolidated_data = {
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'news': {
                        'finnhub_news': news_data,
                        'news_count': len(news_data) if news_data else 0,
                        'sentiment_score': sentiment_analysis['overall_sentiment'],
                        'sentiment_distribution': sentiment_analysis['distribution']
                    },
                    'social_sentiment': {
                        'reddit_sentiment': reddit_sentiment,
                        'social_volume': reddit_sentiment.get('mention_count', 0),
                        'social_score': reddit_sentiment.get('sentiment_score', 0)
                    },
                    'combined_sentiment': {
                        'overall_score': (sentiment_analysis['overall_sentiment'] + reddit_sentiment.get('sentiment_score', 0)) / 2,
                        'confidence': sentiment_analysis.get('confidence', 0.5),
                        'trend': self._determine_sentiment_trend(sentiment_analysis, reddit_sentiment)
                    }
                }
                
                # Sauvegarde
                file_path = self.data_path / f'sentiment_analysis/{symbol}_sentiment_{timestamp}.json'
                with open(file_path, 'w') as f:
                    json.dump(consolidated_data, f, indent=2, default=str)
                
                logger.info(f"📊 Sentiment {symbol}: {consolidated_data['combined_sentiment']['overall_score']:.2f}")
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"❌ Erreur collecte sentiment {symbol}: {e}")
    
    def _get_finnhub_news(self, symbol):
        """Récupère les news depuis Finnhub"""
        try:
            if not self.finnhub_client:
                return []
                
            # News des dernières 24h
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)
            
            news = self.finnhub_client.company_news(
                symbol, 
                _from=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d')
            )
            
            return news[:20]  # Limite à 20 articles
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération news Finnhub {symbol}: {e}")
            return []
    
    def _analyze_news_sentiment(self, news_data):
        """Analyse le sentiment des articles de news"""
        if not news_data:
            return {'overall_sentiment': 0, 'distribution': {'positive': 0, 'neutral': 0, 'negative': 0}, 'confidence': 0}
        
        sentiments = []
        sentiment_scores = {'positive': 0, 'neutral': 0, 'negative': 0}
        
        for article in news_data:
            try:
                # Analyse du titre et du résumé
                text = f"{article.get('headline', '')} {article.get('summary', '')}"
                
                if text.strip():
                    blob = TextBlob(text)
                    polarity = blob.sentiment.polarity
                    
                    sentiments.append(polarity)
                    
                    if polarity > 0.1:
                        sentiment_scores['positive'] += 1
                    elif polarity < -0.1:
                        sentiment_scores['negative'] += 1
                    else:
                        sentiment_scores['neutral'] += 1
                        
            except Exception as e:
                logger.warning(f"⚠️ Erreur analyse sentiment article: {e}")
        
        if sentiments:
            overall_sentiment = np.mean(sentiments)
            confidence = 1 - np.std(sentiments) if len(sentiments) > 1 else 0.5
        else:
            overall_sentiment = 0
            confidence = 0
        
        total_articles = len(news_data)
        distribution = {
            'positive': sentiment_scores['positive'] / total_articles if total_articles > 0 else 0,
            'neutral': sentiment_scores['neutral'] / total_articles if total_articles > 0 else 0,
            'negative': sentiment_scores['negative'] / total_articles if total_articles > 0 else 0
        }
        
        return {
            'overall_sentiment': overall_sentiment,
            'distribution': distribution,
            'confidence': confidence,
            'article_count': total_articles
        }
    
    def _scrape_reddit_sentiment(self, symbol):
        """Scrape le sentiment Reddit (simulation - remplacez par API Reddit réelle)"""
        try:
            # Simulation de données Reddit
            reddit_data = {
                'symbol': symbol,
                'mention_count': np.random.randint(50, 500),
                'sentiment_score': np.random.uniform(-0.5, 0.5),
                'upvote_ratio': np.random.uniform(0.4, 0.9),
                'comment_volume': np.random.randint(100, 1000),
                'trending_keywords': ['bullish', 'moon', 'diamond hands', 'hodl', 'dip'],
                'subreddits': ['wallstreetbets', 'stocks', 'investing']
            }
            return reddit_data
        except Exception as e:
            logger.error(f"❌ Erreur scraping Reddit pour {symbol}: {e}")
            return {
                'mention_count': 0,
                'sentiment_score': 0
            }

    def _determine_sentiment_trend(self, news_sentiment, reddit_sentiment):
        """Détermine la tendance du sentiment global"""
        try:
            combined_score = (news_sentiment['overall_sentiment'] + reddit_sentiment.get('sentiment_score', 0)) / 2
            if combined_score > 0.2:
                return 'Haussier'
            elif combined_score < -0.2:
                return 'Baissier'
            else:
                return 'Neutre'
        except Exception as e:
            logger.error(f"❌ Erreur détermination tendance sentiment: {e}")
            return 'Inconnu'