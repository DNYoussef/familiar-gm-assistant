"""
Data Source Manager
Orchestrates multiple data sources with intelligent routing and failover
"""

import asyncio
from lib.shared.utilities import get_logger
logger = get_logger(__name__)
        self.sources = {}
        self.source_status = {}
        self._initialize_sources()

    def _initialize_sources(self):
        """Initialize all available data sources"""
        for name, source_config in config.data_sources.items():
            if not source_config.enabled:
                continue

            try:
                if name == "alpaca":
                    self.sources[name] = AlpacaSource(source_config)
                    self.source_status[name] = SourceStatus(
                        name=name,
                        priority=SourcePriority.SECONDARY
                    )
                elif name == "polygon":
                    self.sources[name] = PolygonSource(source_config)
                    self.source_status[name] = SourceStatus(
                        name=name,
                        priority=SourcePriority.SECONDARY
                    )
                elif name == "yahoo":
                    self.sources[name] = YahooSource(source_config)
                    self.source_status[name] = SourceStatus(
                        name=name,
                        priority=SourcePriority.PRIMARY  # Yahoo is free and reliable
                    )

                self.logger.info(f"Initialized {name} data source")

            except Exception as e:
                self.logger.error(f"Failed to initialize {name}: {e}")
                self.source_status[name] = SourceStatus(
                    name=name,
                    available=False,
                    last_error=str(e),
                    priority=SourcePriority.DISABLED
                )

    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1D",
        preferred_sources: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get historical data with intelligent source selection

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
            preferred_sources: List of preferred sources to try first

        Returns:
            DataFrame with historical data
        """
        # Determine source order
        sources_to_try = self._get_source_order(preferred_sources)

        for source_name in sources_to_try:
            if not self._is_source_available(source_name):
                continue

            try:
                start_time = datetime.now()
                source = self.sources[source_name]

                self.logger.debug(f"Trying {source_name} for {symbol}")

                data = await source.get_historical_data(
                    symbol, start_date, end_date, timeframe
                )

                response_time = (datetime.now() - start_time).total_seconds()

                if data is not None and not data.empty:
                    # Update source status on success
                    self._update_source_status(source_name, True, response_time)

                    # Validate data quality
                    if self._validate_data_quality(data, symbol, start_date, end_date):
                        self.logger.info(
                            f"Successfully retrieved {len(data)} records for {symbol} from {source_name}"
                        )
                        return data
                    else:
                        self.logger.warning(f"Data quality check failed for {symbol} from {source_name}")

                else:
                    self.logger.warning(f"No data returned for {symbol} from {source_name}")

            except Exception as e:
                self.logger.warning(f"{source_name} failed for {symbol}: {e}")
                self._update_source_status(source_name, False, 0.0, str(e))

        self.logger.error(f"All sources failed for {symbol}")
        return None

    async def get_real_time_quotes(
        self,
        symbols: List[str],
        preferred_sources: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Get real-time quotes for multiple symbols"""
        quotes = {}
        sources_to_try = self._get_source_order(preferred_sources)

        # Create tasks for parallel processing
        tasks = []
        for symbol in symbols:
            for source_name in sources_to_try:
                if self._is_source_available(source_name):
                    task = asyncio.create_task(
                        self._get_quote_from_source(symbol, source_name)
                    )
                    tasks.append((symbol, source_name, task))
                    break  # Only try first available source per symbol

        # Collect results
        for symbol, source_name, task in tasks:
            try:
                quote = await task
                if quote:
                    quotes[symbol] = quote
                    self.logger.debug(f"Got quote for {symbol} from {source_name}")
            except Exception as e:
                self.logger.warning(f"Quote failed for {symbol} from {source_name}: {e}")

        return quotes

    async def _get_quote_from_source(
        self,
        symbol: str,
        source_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get quote from specific source"""
        try:
            source = self.sources[source_name]
            quote = await source.get_real_time_quote(symbol)

            if quote:
                self._update_source_status(source_name, True)
                return quote

        except Exception as e:
            self._update_source_status(source_name, False, 0.0, str(e))

        return None

    def _get_source_order(self, preferred_sources: Optional[List[str]] = None) -> List[str]:
        """Get optimal source order based on priority and availability"""
        available_sources = []

        # Add preferred sources first
        if preferred_sources:
            for source in preferred_sources:
                if self._is_source_available(source):
                    available_sources.append(source)

        # Add remaining sources by priority
        remaining_sources = [
            (name, status.priority.value, status.response_time)
            for name, status in self.source_status.items()
            if self._is_source_available(name) and name not in available_sources
        ]

        # Sort by priority (lower number = higher priority) then response time
        remaining_sources.sort(key=lambda x: (x[1], x[2]))

        for name, _, _ in remaining_sources:
            available_sources.append(name)

        return available_sources

    def _is_source_available(self, source_name: str) -> bool:
        """Check if source is available"""
        if source_name not in self.source_status:
            return False

        status = self.source_status[source_name]

        # Disable source if too many consecutive errors
        if status.error_count > 5:
            return False

        return status.available and status.priority != SourcePriority.DISABLED

    def _update_source_status(
        self,
        source_name: str,
        success: bool,
        response_time: float = 0.0,
        error_message: Optional[str] = None
    ):
        """Update source status based on operation result"""
        if source_name not in self.source_status:
            return

        status = self.source_status[source_name]

        if success:
            status.available = True
            status.last_success = datetime.now()
            status.error_count = 0
            status.last_error = None

            # Update response time with exponential moving average
            if status.response_time == 0:
                status.response_time = response_time
            else:
                status.response_time = 0.7 * status.response_time + 0.3 * response_time

        else:
            status.error_count += 1
            status.last_error = error_message

            # Disable source temporarily if too many errors
            if status.error_count >= 3:
                status.available = False
                self.logger.warning(f"Temporarily disabled {source_name} due to errors")

    def _validate_data_quality(
        self,
        data: pd.DataFrame,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> bool:
        """Validate data quality"""
        if data.empty:
            return False

        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.logger.warning(f"Missing columns for {symbol}: {missing_columns}")
            return False

        # Check data integrity
        invalid_ohlc = (data['high'] < data['low']).any()
        if invalid_ohlc:
            self.logger.warning(f"Invalid OHLC data for {symbol}")
            return False

        # Check for reasonable data coverage
        expected_days = (end_date - start_date).days
        actual_days = len(data)

        # For daily data, expect at least 60% coverage (accounting for weekends/holidays)
        if expected_days > 0:
            coverage_ratio = actual_days / expected_days
            if coverage_ratio < 0.6:
                self.logger.warning(f"Low data coverage for {symbol}: {coverage_ratio:.2f}")
                return False

        return True

    def get_source_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all sources"""
        health = {}

        for name, status in self.source_status.items():
            health[name] = {
                "available": status.available,
                "priority": status.priority.name,
                "error_count": status.error_count,
                "last_error": status.last_error,
                "last_success": status.last_success.isoformat() if status.last_success else None,
                "response_time": round(status.response_time, 3)
            }

        return health

    async def close_all_sources(self):
        """Close all source connections"""
        for source in self.sources.values():
            try:
                await source.close()
            except Exception as e:
                self.logger.warning(f"Error closing source: {e}")

    def __del__(self):
        """Cleanup on destruction"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.close_all_sources())
        except Exception:
            pass