"""Financial Sentiment Analyzer

High-performance sentiment analysis for financial texts using specialized BERT.
Optimized for real-time market sentiment processing with <100ms inference.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
import numpy as np
from typing import Dict, List, Optional, Union, Any
import time
import json
from dataclasses import dataclass
from .financial_bert import FinancialBERT
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import threading


@dataclass
class SentimentConfig:
    """Configuration for sentiment analyzer."""
    model_name: str = 'bert-base-uncased'
    max_length: int = 256  # Reduced for faster inference
    batch_size: int = 16
    num_classes: int = 3
    cache_size: int = 1000
    inference_timeout: float = 0.1  # 100ms

    # Financial sentiment thresholds
    positive_threshold: float = 0.6
    negative_threshold: float = 0.6
    neutral_threshold: float = 0.4

    # Market impact weights
    news_weight: float = 0.4
    social_weight: float = 0.3
    earnings_weight: float = 0.8
    analyst_weight: float = 0.7

    # Gary×Taleb integration
    dpi_sentiment_factor: float = 0.2
    antifragile_contrarian_boost: float = 0.15


class FinancialSentimentAnalyzer:
    """Real-time financial sentiment analyzer.

    Processes news, social media, earnings calls, and analyst reports
    with <100ms inference time and GaryTaleb integration.
    """

    def __init__(self, config: SentimentConfig):
        """Initialize sentiment analyzer.

        Args:
            config: Sentiment analyzer configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model and tokenizer
        self.model = FinancialBERT(
            model_name=config.model_name,
            num_classes=config.num_classes,
            max_length=config.max_length
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # Optimize for inference
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Compile model if PyTorch 2.0+
        if torch.__version__ >= "2.0.0":
            self.model = torch.compile(self.model, mode='max-autotune')

        # Cache for repeated queries
        self.sentiment_cache = {}
        self.cache_lock = threading.Lock()

        # Performance tracking
        self.inference_times = []
        self.processing_stats = {
            'total_processed': 0,
            'cache_hits': 0,
            'avg_confidence': 0.0,
            'market_impact_events': 0
        }

        # Async processing pool
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Financial text preprocessing patterns
        self.preprocessing_patterns = self._setup_preprocessing()

    def _setup_preprocessing(self) -> Dict[str, str]:
        """Setup text preprocessing patterns for financial content."""
        return {
            # Stock symbol normalization
            r'\$([A-Z]{1,5})': r'STOCK_\1',
            r'\b([A-Z]{1,5})\b(?=\s(?:stock|shares|equity))': r'STOCK_\1',

            # Number normalization
            r'\$[\d,]+\.?\d*[BMK]?': 'CURRENCY_AMOUNT',
            r'\d+\.?\d*%': 'PERCENTAGE',
            r'\b\d{4}\b(?=\s(?:Q[1-4]|quarter|year))': 'YEAR',

            # Time normalization
            r'\b(?:today|yesterday|tomorrow)\b': 'TIME_REF',
            r'\b(?:this|next|last)\s(?:week|month|quarter|year)\b': 'TIME_PERIOD',

            # Market events
            r'\b(?:earnings|guidance|conference call)\b': 'EARNINGS_EVENT',
            r'\b(?:merger|acquisition|IPO|buyback)\b': 'CORPORATE_EVENT',
            r'\b(?:FDA approval|regulatory|lawsuit)\b': 'REGULATORY_EVENT'
        }

    def preprocess_text(self, text: str) -> str:
        """Preprocess financial text for better analysis.

        Args:
            text: Raw input text

        Returns:
            Preprocessed text optimized for sentiment analysis
        """
        import re

        # Apply preprocessing patterns
        processed_text = text
        for pattern, replacement in self.preprocessing_patterns.items():
            processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)

        # Remove excessive whitespace and normalize
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()

        return processed_text

    def analyze_sentiment(self,
                         text: str,
                         source_type: str = 'news',
                         use_cache: bool = True) -> Dict[str, Any]:
        """Analyze sentiment of financial text.

        Args:
            text: Input text
            source_type: Type of source (news, social, earnings, analyst)
            use_cache: Whether to use caching

        Returns:
            Comprehensive sentiment analysis results
        """
        start_time = time.time()

        # Check cache first
        if use_cache:
            cache_key = hash(text + source_type)
            with self.cache_lock:
                if cache_key in self.sentiment_cache:
                    self.processing_stats['cache_hits'] += 1
                    cached_result = self.sentiment_cache[cache_key].copy()
                    cached_result['inference_time_ms'] = (time.time() - start_time) * 1000
                    cached_result['from_cache'] = True
                    return cached_result

        # Preprocess text
        processed_text = self.preprocess_text(text)

        # Get model predictions
        predictions = self.model.predict_sentiment(processed_text, self.tokenizer)

        # Apply source-specific weighting
        source_weights = {
            'news': self.config.news_weight,
            'social': self.config.social_weight,
            'earnings': self.config.earnings_weight,
            'analyst': self.config.analyst_weight
        }

        source_weight = source_weights.get(source_type, 1.0)

        # Calculate weighted sentiment scores
        weighted_sentiment = predictions['overall_sentiment'] * source_weight

        # Gary's DPI integration
        dpi_adjustment = self._calculate_dpi_sentiment_factor(text, predictions)

        # Taleb's antifragility enhancement
        antifragile_adjustment = self._apply_antifragile_sentiment(predictions, source_type)

        # Final sentiment calculation
        final_sentiment = (
            weighted_sentiment +
            dpi_adjustment * self.config.dpi_sentiment_factor +
            antifragile_adjustment * self.config.antifragile_contrarian_boost
        )

        # Market impact assessment
        market_impact = self._assess_market_impact(predictions, source_type, final_sentiment)

        # Trading signal generation
        trading_signal = self._generate_trading_signal(final_sentiment, predictions, market_impact)

        # Compile results
        result = {
            'raw_sentiment': predictions['overall_sentiment'],
            'weighted_sentiment': weighted_sentiment,
            'final_sentiment': final_sentiment,
            'confidence': predictions['confidence'],
            'predicted_class': predictions['predicted_class'],
            'sentiment_scores': predictions['sentiment_scores'],
            'volatility_impact': predictions['volatility_impact'],
            'urgency_score': predictions['urgency_score'],
            'market_impact': market_impact,
            'trading_signal': trading_signal,
            'source_type': source_type,
            'source_weight': source_weight,
            'dpi_adjustment': dpi_adjustment,
            'antifragile_adjustment': antifragile_adjustment,
            'financial_terms': predictions['financial_terms_detected'],
            'processed_text': processed_text,
            'inference_time_ms': (time.time() - start_time) * 1000,
            'from_cache': False,
            'gary_dpi_integrated': True,
            'taleb_antifragile_enhanced': True
        }

        # Update cache
        if use_cache and len(self.sentiment_cache) < self.config.cache_size:
            with self.cache_lock:
                self.sentiment_cache[cache_key] = result.copy()

        # Update statistics
        self.processing_stats['total_processed'] += 1
        self.processing_stats['avg_confidence'] = (
            (self.processing_stats['avg_confidence'] * (self.processing_stats['total_processed'] - 1) +
             predictions['confidence']) / self.processing_stats['total_processed']
        )

        if abs(final_sentiment) > 0.7:  # High impact event
            self.processing_stats['market_impact_events'] += 1

        self.inference_times.append(result['inference_time_ms'])

        return result

    def _calculate_dpi_sentiment_factor(self, text: str, predictions: Dict) -> float:
        """Calculate Gary's DPI sentiment factor.

        Args:
            text: Original text
            predictions: Model predictions

        Returns:
            DPI sentiment adjustment factor
        """
        # Gary's Dynamic Position Intelligence for sentiment
        dpi_keywords = {
            'momentum': 0.3,
            'breakout': 0.4,
            'volume': 0.2,
            'institutional': 0.3,
            'technical analysis': 0.2,
            'support': 0.1,
            'resistance': -0.1,
            'trend': 0.2
        }

        dpi_factor = 0.0
        text_lower = text.lower()

        for keyword, weight in dpi_keywords.items():
            if keyword in text_lower:
                dpi_factor += weight * predictions['confidence']

        # Normalize and bound
        dpi_factor = np.tanh(dpi_factor)  # Keep in [-1, 1]

        return dpi_factor

    def _apply_antifragile_sentiment(self, predictions: Dict, source_type: str) -> float:
        """Apply Taleb's antifragility principles to sentiment.

        Args:
            predictions: Model predictions
            source_type: Source type

        Returns:
            Antifragile sentiment adjustment
        """
        # Antifragile enhancement: benefit from disorder and volatility
        volatility = predictions['volatility_impact']
        urgency = predictions['urgency_score']

        # Higher volatility = more opportunity (contrarian approach)
        antifragile_factor = volatility * 0.5 + urgency * 0.3

        # Source-specific antifragile weighting
        source_multipliers = {
            'social': 1.2,  # Social sentiment often overreacts
            'news': 0.8,    # News is more measured
            'earnings': 0.5,  # Earnings are factual
            'analyst': 0.6   # Analyst reports are professional
        }

        multiplier = source_multipliers.get(source_type, 1.0)

        # Apply contrarian logic: when everyone is very positive/negative, be cautious
        extreme_sentiment = abs(predictions['overall_sentiment'])
        if extreme_sentiment > 0.8:
            antifragile_factor *= -0.5 * extreme_sentiment  # Contrarian adjustment

        return antifragile_factor * multiplier

    def _assess_market_impact(self,
                            predictions: Dict,
                            source_type: str,
                            final_sentiment: float) -> Dict[str, float]:
        """Assess potential market impact of sentiment.

        Args:
            predictions: Model predictions
            source_type: Source type
            final_sentiment: Final adjusted sentiment

        Returns:
            Market impact assessment
        """
        base_impact = predictions['market_impact_probs']

        # Source impact multipliers
        impact_multipliers = {
            'earnings': 2.0,
            'analyst': 1.5,
            'news': 1.2,
            'social': 0.8
        }

        multiplier = impact_multipliers.get(source_type, 1.0)

        # Calculate expected price impact
        price_impact = final_sentiment * predictions['confidence'] * multiplier

        # Time decay factor (urgent news has immediate impact)
        time_decay = 1.0 - (1.0 - predictions['urgency_score']) * 0.3

        return {
            'short_term_impact': price_impact * time_decay,
            'medium_term_impact': price_impact * 0.7,
            'long_term_impact': price_impact * 0.4,
            'volatility_impact': predictions['volatility_impact'] * multiplier,
            'confidence': predictions['confidence'],
            'source_multiplier': multiplier,
            'time_decay_factor': time_decay
        }

    def _generate_trading_signal(self,
                               final_sentiment: float,
                               predictions: Dict,
                               market_impact: Dict) -> Dict[str, Any]:
        """Generate trading signal based on sentiment analysis.

        Args:
            final_sentiment: Final sentiment score
            predictions: Model predictions
            market_impact: Market impact assessment

        Returns:
            Trading signal with confidence and timing
        """
        # Signal strength based on sentiment and confidence
        signal_strength = abs(final_sentiment) * predictions['confidence']

        # Determine signal direction
        if final_sentiment > self.config.positive_threshold:
            signal = 'BUY'
            direction = 1
        elif final_sentiment < -self.config.negative_threshold:
            signal = 'SELL'
            direction = -1
        else:
            signal = 'HOLD'
            direction = 0

        # Signal timing based on urgency
        if predictions['urgency_score'] > 0.7:
            timing = 'IMMEDIATE'
        elif predictions['urgency_score'] > 0.4:
            timing = 'NEAR_TERM'
        else:
            timing = 'LONG_TERM'

        # Risk assessment
        risk_level = predictions['volatility_impact']
        if risk_level > 0.8:
            risk = 'HIGH'
        elif risk_level > 0.5:
            risk = 'MEDIUM'
        else:
            risk = 'LOW'

        return {
            'signal': signal,
            'direction': direction,
            'strength': signal_strength,
            'timing': timing,
            'risk_level': risk,
            'confidence': predictions['confidence'],
            'expected_impact': market_impact['short_term_impact'],
            'hold_duration': self._estimate_hold_duration(predictions),
            'stop_loss_level': self._calculate_stop_loss(final_sentiment, risk_level),
            'take_profit_level': self._calculate_take_profit(final_sentiment, signal_strength)
        }

    def _estimate_hold_duration(self, predictions: Dict) -> str:
        """Estimate optimal hold duration based on sentiment characteristics."""
        urgency = predictions['urgency_score']
        volatility = predictions['volatility_impact']

        if urgency > 0.8:
            return 'MINUTES'  # Very urgent, act quickly
        elif urgency > 0.6:
            return 'HOURS'    # Urgent, day trading
        elif volatility > 0.7:
            return 'DAYS'     # Volatile, short-term swing
        else:
            return 'WEEKS'    # Stable, longer-term position

    def _calculate_stop_loss(self, sentiment: float, risk: float) -> float:
        """Calculate stop loss level based on sentiment and risk."""
        base_stop = 0.02  # 2% base stop loss

        # Adjust based on sentiment strength
        sentiment_adjustment = min(abs(sentiment) * 0.01, 0.01)

        # Adjust based on risk level
        risk_adjustment = risk * 0.015

        return base_stop + sentiment_adjustment + risk_adjustment

    def _calculate_take_profit(self, sentiment: float, strength: float) -> float:
        """Calculate take profit level based on sentiment and signal strength."""
        base_profit = 0.04  # 4% base take profit

        # Adjust based on sentiment and strength
        profit_multiplier = 1 + (strength * 0.5)

        return base_profit * profit_multiplier

    def batch_analyze(self,
                     texts: List[str],
                     source_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Analyze multiple texts in batch for efficiency.

        Args:
            texts: List of texts to analyze
            source_types: List of source types (optional)

        Returns:
            List of sentiment analysis results
        """
        if source_types is None:
            source_types = ['news'] * len(texts)

        # Process in batches to optimize GPU utilization
        batch_size = self.config.batch_size
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_sources = source_types[i:i + batch_size]

            batch_results = []
            for text, source in zip(batch_texts, batch_sources):
                result = self.analyze_sentiment(text, source)
                batch_results.append(result)

            results.extend(batch_results)

        return results

    async def analyze_sentiment_async(self,
                                    text: str,
                                    source_type: str = 'news') -> Dict[str, Any]:
        """Asynchronously analyze sentiment for real-time processing.

        Args:
            text: Input text
            source_type: Source type

        Returns:
            Sentiment analysis results
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self.analyze_sentiment,
            text,
            source_type
        )
        return result

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the sentiment analyzer."""
        if not self.inference_times:
            return {"status": "No inference data available"}

        recent_times = self.inference_times[-100:]  # Last 100 inferences

        return {
            'avg_inference_time_ms': np.mean(recent_times),
            'max_inference_time_ms': np.max(recent_times),
            'min_inference_time_ms': np.min(recent_times),
            'inference_target_met': np.mean(recent_times) < 100,
            'total_processed': self.processing_stats['total_processed'],
            'cache_hit_rate': (self.processing_stats['cache_hits'] /
                             max(self.processing_stats['total_processed'], 1)),
            'avg_confidence': self.processing_stats['avg_confidence'],
            'market_impact_events': self.processing_stats['market_impact_events'],
            'cache_size': len(self.sentiment_cache),
            'gpu_available': torch.cuda.is_available(),
            'model_parameters': sum(p.numel() for p in self.model.parameters())
        }

    def clear_cache(self):
        """Clear sentiment analysis cache."""
        with self.cache_lock:
            self.sentiment_cache.clear()

    def export_model(self, path: str):
        """Export trained model for deployment.

        Args:
            path: Export path
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'preprocessing_patterns': self.preprocessing_patterns,
            'performance_stats': self.processing_stats
        }, path)

    def load_model(self, path: str):
        """Load trained model from checkpoint.

        Args:
            path: Model checkpoint path
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.processing_stats = checkpoint.get('performance_stats', self.processing_stats)


# Factory function for easy creation
def create_sentiment_analyzer(
    max_length: int = 256,
    enable_caching: bool = True,
    dpi_integration: bool = True,
    antifragile_enhancement: bool = True
) -> FinancialSentimentAnalyzer:
    """Create financial sentiment analyzer with GaryTaleb integration.

    Args:
        max_length: Maximum sequence length for faster processing
        enable_caching: Enable result caching
        dpi_integration: Enable Gary's DPI calculations
        antifragile_enhancement: Enable Taleb's antifragility principles

    Returns:
        Configured sentiment analyzer
    """
    config = SentimentConfig(
        max_length=max_length,
        cache_size=1000 if enable_caching else 0,
        dpi_sentiment_factor=0.2 if dpi_integration else 0.0,
        antifragile_contrarian_boost=0.15 if antifragile_enhancement else 0.0
    )

    analyzer = FinancialSentimentAnalyzer(config)
    print(f\"Financial Sentiment Analyzer created with GaryTaleb integration\")
    print(f\"Target inference time: <100ms\")

    return analyzer"