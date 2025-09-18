"""
Historical Data Loader
High-performance loader supporting multiple data sources with parallel processing
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from lib.shared.utilities import get_logger
logger = get_logger(__name__)
        self.sources = self._initialize_sources()
        self.executor = ThreadPoolExecutor(max_workers=config.processing.processing_threads)

    def _initialize_sources(self) -> Dict[str, Any]:
        """Initialize data source connectors"""
        sources = {}

        # Initialize enabled sources
        for name, source_config in config.data_sources.items():
            if not source_config.enabled:
                continue

            try:
                if name == "alpaca":
                    sources[name] = AlpacaSource(source_config)
                elif name == "polygon":
                    sources[name] = PolygonSource(source_config)
                elif name == "yahoo":
                    sources[name] = YahooSource(source_config)

                self.logger.info(f"Initialized {name} data source")
            except Exception as e:
                self.logger.warning(f"Failed to initialize {name} source: {e}")

        return sources

    async def load_historical_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1D",
        source: str = "yahoo"
    ) -> Dict[str, pd.DataFrame]:
        """
        Load historical data for multiple symbols

        Args:
            symbols: List of stock symbols
            start_date: Start date for data
            end_date: End date for data
            timeframe: Data timeframe (1Min, 5Min, 15Min, 1H, 1D)
            source: Primary data source

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        self.logger.info(f"Loading historical data for {len(symbols)} symbols from {source}")

        # Create data requests
        requests = [
            DataRequest(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
                source=source
            )
            for symbol in symbols
        ]

        # Process requests in parallel
        results = {}
        tasks = []

        for request in requests:
            task = asyncio.create_task(self._load_single_symbol(request))
            tasks.append((request.symbol, task))

        # Collect results
        for symbol, task in tasks:
            try:
                data = await task
                if data is not None and not data.empty:
                    results[symbol] = data
                    self.logger.info(f"Successfully loaded {len(data)} records for {symbol}")
                else:
                    self.logger.warning(f"No data received for {symbol}")
            except Exception as e:
                self.logger.error(f"Failed to load data for {symbol}: {e}")

        return results

    async def _load_single_symbol(self, request: DataRequest) -> Optional[pd.DataFrame]:
        """Load data for a single symbol with failover"""
        sources_to_try = [request.source] + request.fallback_sources

        for source_name in sources_to_try:
            if source_name not in self.sources:
                continue

            try:
                source = self.sources[source_name]
                data = await source.get_historical_data(
                    request.symbol,
                    request.start_date,
                    request.end_date,
                    request.timeframe
                )

                if data is not None and not data.empty:
                    # Validate and clean data
                    cleaned_data = self._clean_data(data, request.symbol)
                    if self._validate_data_quality(cleaned_data):
                        return cleaned_data

            except Exception as e:
                self.logger.warning(f"Source {source_name} failed for {request.symbol}: {e}")
                continue

        self.logger.error(f"All sources failed for {request.symbol}")
        return None

    def _clean_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean and normalize data"""
        if data.empty:
            return data

        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.logger.warning(f"Missing columns for {symbol}: {missing_columns}")
            return data

        # Remove rows with invalid data
        data = data.dropna(subset=required_columns)

        # Remove rows where high < low (data integrity check)
        invalid_rows = data['high'] < data['low']
        if invalid_rows.any():
            self.logger.warning(f"Removed {invalid_rows.sum()} rows with high < low for {symbol}")
            data = data[~invalid_rows]

        # Remove rows with zero or negative prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            invalid_prices = data[col] <= 0
            if invalid_prices.any():
                self.logger.warning(f"Removed {invalid_prices.sum()} rows with invalid {col} prices for {symbol}")
                data = data[~invalid_prices]

        # Remove rows with negative volume
        negative_volume = data['volume'] < 0
        if negative_volume.any():
            self.logger.warning(f"Removed {negative_volume.sum()} rows with negative volume for {symbol}")
            data = data[~negative_volume]

        # Sort by timestamp
        if data.index.name == 'timestamp' or 'timestamp' in data.columns:
            data = data.sort_index()

        # Add symbol column
        data['symbol'] = symbol

        return data

    def _validate_data_quality(self, data: pd.DataFrame) -> bool:
        """Validate data quality"""
        if data.empty:
            return False

        # Check completeness
        completeness = 1.0 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
        if completeness < config.validation.completeness_threshold:
            self.logger.warning(f"Data completeness {completeness:.2f} below threshold")
            return False

        return True

    def load_batch_data(
        self,
        symbol_batches: List[List[str]],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1D"
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data in batches for memory efficiency

        Args:
            symbol_batches: List of symbol batches
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe

        Returns:
            Combined results from all batches
        """
        all_results = {}

        for i, batch in enumerate(symbol_batches):
            self.logger.info(f"Processing batch {i+1}/{len(symbol_batches)} with {len(batch)} symbols")

            # Use asyncio to run the async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                batch_results = loop.run_until_complete(
                    self.load_historical_data(batch, start_date, end_date, timeframe)
                )
                all_results.update(batch_results)
            finally:
                loop.close()

        return all_results

    def get_data_coverage(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        source: str = "yahoo"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get data coverage information for symbols

        Returns:
            Coverage statistics for each symbol
        """
        coverage = {}

        for symbol in symbols:
            try:
                if source in self.sources:
                    source_obj = self.sources[source]
                    # Get sample data to check availability
                    sample_data = asyncio.run(source_obj.get_historical_data(
                        symbol, start_date, start_date + timedelta(days=7), "1D"
                    ))

                    if sample_data is not None and not sample_data.empty:
                        coverage[symbol] = {
                            "available": True,
                            "source": source,
                            "sample_records": len(sample_data),
                            "earliest_date": sample_data.index.min() if len(sample_data) > 0 else None
                        }
                    else:
                        coverage[symbol] = {"available": False, "source": source}

            except Exception as e:
                coverage[symbol] = {"available": False, "error": str(e)}

        return coverage

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)