"""
Real-Time Data Streamer
High-performance streaming system targeting <50ms latency
"""

import asyncio
import websockets
import json
from lib.shared.utilities import get_logger
logger = get_logger(__name__)
        self.subscribers: Dict[str, List[Callable]] = {}
        self.stream_buffer = StreamBuffer()
        self.failover_manager = FailoverManager()

        # Performance tracking
        self.metrics = StreamMetrics()
        self.message_count = 0
        self.latency_samples = deque(maxlen=1000)
        self.last_metrics_update = time.time()

        # Connection management
        self.active_connections: Dict[str, Any] = {}
        self.subscribed_symbols: Set[str] = set()

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.processing_loop = None
        self.metrics_loop = None
        self.running = False

    async def start(self):
        """Start the streaming system"""
        self.logger.info("Starting real-time streaming system")
        self.running = True

        # Start background tasks
        self.processing_loop = asyncio.create_task(self._processing_loop())
        self.metrics_loop = asyncio.create_task(self._metrics_loop())

        # Initialize connections
        await self._initialize_connections()

    async def stop(self):
        """Stop the streaming system"""
        self.logger.info("Stopping real-time streaming system")
        self.running = False

        # Stop background tasks
        if self.processing_loop:
            self.processing_loop.cancel()
        if self.metrics_loop:
            self.metrics_loop.cancel()

        # Close all connections
        await self._close_all_connections()

    async def subscribe_symbols(self, symbols: List[str], data_types: List[str] = None):
        """
        Subscribe to real-time data for symbols

        Args:
            symbols: List of symbols to subscribe to
            data_types: Types of data to subscribe to ["quotes", "trades", "bars"]
        """
        if data_types is None:
            data_types = ["quotes", "trades"]

        self.subscribed_symbols.update(symbols)

        for connection_name, connection in self.active_connections.items():
            try:
                await self._send_subscription(connection, symbols, data_types, "subscribe")
                self.logger.info(f"Subscribed to {len(symbols)} symbols on {connection_name}")
            except Exception as e:
                self.logger.error(f"Failed to subscribe on {connection_name}: {e}")

    async def unsubscribe_symbols(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        self.subscribed_symbols -= set(symbols)

        for connection_name, connection in self.active_connections.items():
            try:
                await self._send_subscription(connection, symbols, [], "unsubscribe")
                self.logger.info(f"Unsubscribed from {len(symbols)} symbols on {connection_name}")
            except Exception as e:
                self.logger.error(f"Failed to unsubscribe on {connection_name}: {e}")

    def add_subscriber(self, callback: Callable[[StreamData], None], symbols: List[str] = None):
        """
        Add callback for streaming data

        Args:
            callback: Function to call with StreamData
            symbols: Optional list of symbols to filter (None = all symbols)
        """
        key = f"all_symbols" if symbols is None else "|".join(sorted(symbols))

        if key not in self.subscribers:
            self.subscribers[key] = []

        self.subscribers[key].append(callback)
        self.logger.info(f"Added subscriber for {key}")

    def remove_subscriber(self, callback: Callable, symbols: List[str] = None):
        """Remove subscriber callback"""
        key = f"all_symbols" if symbols is None else "|".join(sorted(symbols))

        if key in self.subscribers and callback in self.subscribers[key]:
            self.subscribers[key].remove(callback)
            if not self.subscribers[key]:
                del self.subscribers[key]
            self.logger.info(f"Removed subscriber for {key}")

    async def _initialize_connections(self):
        """Initialize WebSocket connections to data sources"""
        # Alpaca WebSocket
        if "alpaca" in config.data_sources and config.data_sources["alpaca"].enabled:
            await self._connect_alpaca()

        # Polygon WebSocket
        if "polygon" in config.data_sources and config.data_sources["polygon"].enabled:
            await self._connect_polygon()

        # Add more sources as needed

    async def _connect_alpaca(self):
        """Connect to Alpaca WebSocket"""
        try:
            alpaca_config = config.data_sources["alpaca"]
            ws_url = "wss://stream.data.alpaca.markets/v2/iex"

            # Create WebSocket connection
            connection = await websockets.connect(
                ws_url,
                extra_headers={"User-Agent": "GaryTaleb-Pipeline/1.0"}
            )

            # Authenticate
            auth_message = {
                "action": "auth",
                "key": alpaca_config.api_key,
                "secret": self._get_alpaca_secret()
            }
            await connection.send(json.dumps(auth_message))

            # Start message handler
            asyncio.create_task(self._handle_alpaca_messages(connection))

            self.active_connections["alpaca"] = connection
            self.logger.info("Connected to Alpaca WebSocket")

        except Exception as e:
            self.logger.error(f"Failed to connect to Alpaca WebSocket: {e}")

    async def _connect_polygon(self):
        """Connect to Polygon WebSocket"""
        try:
            polygon_config = config.data_sources["polygon"]
            ws_url = "wss://socket.polygon.io/stocks"

            connection = await websockets.connect(ws_url)

            # Authenticate
            auth_message = {
                "action": "auth",
                "params": polygon_config.api_key
            }
            await connection.send(json.dumps(auth_message))

            # Start message handler
            asyncio.create_task(self._handle_polygon_messages(connection))

            self.active_connections["polygon"] = connection
            self.logger.info("Connected to Polygon WebSocket")

        except Exception as e:
            self.logger.error(f"Failed to connect to Polygon WebSocket: {e}")

    async def _handle_alpaca_messages(self, connection):
        """Handle Alpaca WebSocket messages"""
        try:
            async for message in connection:
                receive_time = time.time()

                try:
                    data = json.loads(message)

                    if isinstance(data, list):
                        for item in data:
                            await self._process_alpaca_message(item, receive_time)
                    else:
                        await self._process_alpaca_message(data, receive_time)

                except json.JSONDecodeError as e:
                    self.logger.warning(f"Invalid JSON from Alpaca: {e}")

        except Exception as e:
            self.logger.error(f"Alpaca message handler error: {e}")
            # Trigger reconnection
            asyncio.create_task(self._reconnect_alpaca())

    async def _handle_polygon_messages(self, connection):
        """Handle Polygon WebSocket messages"""
        try:
            async for message in connection:
                receive_time = time.time()

                try:
                    data = json.loads(message)

                    if isinstance(data, list):
                        for item in data:
                            await self._process_polygon_message(item, receive_time)
                    else:
                        await self._process_polygon_message(data, receive_time)

                except json.JSONDecodeError as e:
                    self.logger.warning(f"Invalid JSON from Polygon: {e}")

        except Exception as e:
            self.logger.error(f"Polygon message handler error: {e}")
            # Trigger reconnection
            asyncio.create_task(self._reconnect_polygon())

    async def _process_alpaca_message(self, message: Dict[str, Any], receive_time: float):
        """Process individual Alpaca message"""
        if message.get("T") == "q":  # Quote
            stream_data = StreamData(
                symbol=message["S"],
                timestamp=datetime.fromisoformat(message["t"].replace("Z", "+00:00")),
                data_type="quote",
                data={
                    "bid": float(message["bp"]),
                    "ask": float(message["ap"]),
                    "bid_size": int(message["bs"]),
                    "ask_size": int(message["as"])
                },
                source="alpaca",
                latency_ms=(receive_time - time.time()) * 1000
            )

            await self.stream_buffer.add_data(stream_data)

        elif message.get("T") == "t":  # Trade
            stream_data = StreamData(
                symbol=message["S"],
                timestamp=datetime.fromisoformat(message["t"].replace("Z", "+00:00")),
                data_type="trade",
                data={
                    "price": float(message["p"]),
                    "size": int(message["s"])
                },
                source="alpaca",
                latency_ms=(receive_time - time.time()) * 1000
            )

            await self.stream_buffer.add_data(stream_data)

    async def _process_polygon_message(self, message: Dict[str, Any], receive_time: float):
        """Process individual Polygon message"""
        if message.get("ev") == "Q":  # Quote
            stream_data = StreamData(
                symbol=message["sym"],
                timestamp=datetime.fromtimestamp(message["t"] / 1000),
                data_type="quote",
                data={
                    "bid": float(message["bp"]),
                    "ask": float(message["ap"]),
                    "bid_size": int(message["bs"]),
                    "ask_size": int(message["as"])
                },
                source="polygon",
                latency_ms=(receive_time - time.time()) * 1000
            )

            await self.stream_buffer.add_data(stream_data)

    async def _processing_loop(self):
        """Main processing loop for buffered data"""
        while self.running:
            try:
                # Process buffered data
                data_batch = await self.stream_buffer.get_batch()

                if data_batch:
                    await self._distribute_data(data_batch)

                # Small delay to prevent CPU spinning
                await asyncio.sleep(0.001)  # 1ms

            except Exception as e:
                self.logger.error(f"Processing loop error: {e}")

    async def _distribute_data(self, data_batch: List[StreamData]):
        """Distribute data to subscribers"""
        for data in data_batch:
            # Update metrics
            self.message_count += 1
            self.latency_samples.append(data.latency_ms)

            # Find matching subscribers
            for key, callbacks in self.subscribers.items():
                if key == "all_symbols" or data.symbol in key:
                    for callback in callbacks:
                        try:
                            # Execute callback in thread pool to avoid blocking
                            self.executor.submit(callback, data)
                        except Exception as e:
                            self.logger.warning(f"Subscriber callback error: {e}")

    async def _metrics_loop(self):
        """Calculate and update performance metrics"""
        while self.running:
            try:
                current_time = time.time()
                time_delta = current_time - self.last_metrics_update

                if time_delta >= 1.0:  # Update every second
                    self.metrics.messages_per_second = self.message_count / time_delta

                    if self.latency_samples:
                        self.metrics.average_latency_ms = np.mean(self.latency_samples)
                        self.metrics.max_latency_ms = np.max(self.latency_samples)

                    self.metrics.buffer_utilization = self.stream_buffer.get_utilization()
                    self.metrics.connection_count = len(self.active_connections)

                    # Reset counters
                    self.message_count = 0
                    self.last_metrics_update = current_time

                await asyncio.sleep(1.0)

            except Exception as e:
                self.logger.error(f"Metrics loop error: {e}")

    async def _send_subscription(self, connection, symbols: List[str], data_types: List[str], action: str):
        """Send subscription message to WebSocket"""
        # Implementation depends on the specific WebSocket protocol
        pass

    def _get_alpaca_secret(self) -> str:
        """Get Alpaca secret key"""
        import os
        return os.getenv("ALPACA_SECRET_KEY", "")

    async def _close_all_connections(self):
        """Close all WebSocket connections"""
        for name, connection in self.active_connections.items():
            try:
                await connection.close()
                self.logger.info(f"Closed {name} connection")
            except Exception as e:
                self.logger.warning(f"Error closing {name}: {e}")

        self.active_connections.clear()

    async def _reconnect_alpaca(self):
        """Reconnect to Alpaca WebSocket"""
        await asyncio.sleep(5)  # Wait before reconnecting
        if self.running:
            await self._connect_alpaca()

    async def _reconnect_polygon(self):
        """Reconnect to Polygon WebSocket"""
        await asyncio.sleep(5)  # Wait before reconnecting
        if self.running:
            await self._connect_polygon()

    def get_metrics(self) -> StreamMetrics:
        """Get current streaming metrics"""
        return self.metrics

    def get_buffer_status(self) -> Dict[str, Any]:
        """Get buffer status information"""
        return self.stream_buffer.get_status()

    def __del__(self):
        """Cleanup on destruction"""
        if self.running:
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(self.stop())
            except Exception:
                pass