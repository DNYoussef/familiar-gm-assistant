"""
WebSocket Manager
Advanced WebSocket connection management with automatic reconnection
"""

import asyncio
import websockets
import json
from lib.shared.utilities import get_logger
logger = get_logger(__name__)
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.connection_configs: Dict[str, ConnectionConfig] = {}
        self.connection_metrics: Dict[str, ConnectionMetrics] = {}

        # Message handlers
        self.message_handlers: Dict[str, Callable] = {}
        self.connection_handlers: Dict[str, Callable] = {}

        # Message queuing for offline periods
        self.message_queues: Dict[str, List[Dict[str, Any]]] = {}
        self.max_queue_size = 10000

        # Background tasks
        self.monitor_task = None
        self.heartbeat_tasks: Dict[str, asyncio.Task] = {}

        self.running = False

    async def start(self):
        """Start the WebSocket manager"""
        self.logger.info("Starting WebSocket manager")
        self.running = True

        # Start connection monitoring
        self.monitor_task = asyncio.create_task(self._monitor_connections())

    async def stop(self):
        """Stop the WebSocket manager"""
        self.logger.info("Stopping WebSocket manager")
        self.running = False

        # Cancel monitoring task
        if self.monitor_task:
            self.monitor_task.cancel()

        # Cancel heartbeat tasks
        for task in self.heartbeat_tasks.values():
            task.cancel()
        self.heartbeat_tasks.clear()

        # Close all connections
        await self._close_all_connections()

    async def add_connection(self, config: ConnectionConfig) -> bool:
        """
        Add and establish a WebSocket connection

        Args:
            config: Connection configuration

        Returns:
            True if connection established successfully
        """
        try:
            self.connection_configs[config.name] = config
            self.connection_metrics[config.name] = ConnectionMetrics(
                state=ConnectionState.DISCONNECTED
            )
            self.message_queues[config.name] = []

            # Attempt to connect
            success = await self._connect(config.name)
            if success:
                self.logger.info(f"Successfully added connection: {config.name}")
            else:
                self.logger.error(f"Failed to establish connection: {config.name}")

            return success

        except Exception as e:
            self.logger.error(f"Error adding connection {config.name}: {e}")
            return False

    async def remove_connection(self, connection_name: str):
        """Remove and close a connection"""
        try:
            # Close connection if active
            if connection_name in self.connections:
                await self._close_connection(connection_name)

            # Cancel heartbeat task
            if connection_name in self.heartbeat_tasks:
                self.heartbeat_tasks[connection_name].cancel()
                del self.heartbeat_tasks[connection_name]

            # Clean up
            self.connection_configs.pop(connection_name, None)
            self.connection_metrics.pop(connection_name, None)
            self.message_queues.pop(connection_name, None)

            self.logger.info(f"Removed connection: {connection_name}")

        except Exception as e:
            self.logger.error(f"Error removing connection {connection_name}: {e}")

    async def send_message(self, connection_name: str, message: Dict[str, Any]) -> bool:
        """
        Send message to specific connection

        Args:
            connection_name: Name of the connection
            message: Message to send

        Returns:
            True if sent successfully
        """
        try:
            if connection_name not in self.connections:
                # Queue message for when connection is available
                self._queue_message(connection_name, message)
                return False

            connection = self.connections[connection_name]
            metrics = self.connection_metrics[connection_name]

            # Send message
            message_json = json.dumps(message)
            await connection.send(message_json)

            # Update metrics
            metrics.messages_sent += 1

            return True

        except Exception as e:
            self.logger.error(f"Error sending message to {connection_name}: {e}")
            # Queue message for retry
            self._queue_message(connection_name, message)
            return False

    async def broadcast_message(self, message: Dict[str, Any]) -> Dict[str, bool]:
        """
        Broadcast message to all active connections

        Args:
            message: Message to broadcast

        Returns:
            Dictionary with connection names and success status
        """
        results = {}

        for connection_name in self.connection_configs.keys():
            results[connection_name] = await self.send_message(connection_name, message)

        return results

    def set_message_handler(self, connection_name: str, handler: Callable[[str, Dict], None]):
        """Set message handler for specific connection"""
        self.message_handlers[connection_name] = handler

    def set_connection_handler(self, connection_name: str, handler: Callable[[str, ConnectionState], None]):
        """Set connection state change handler"""
        self.connection_handlers[connection_name] = handler

    async def _connect(self, connection_name: str) -> bool:
        """Establish WebSocket connection"""
        config = self.connection_configs[connection_name]
        metrics = self.connection_metrics[connection_name]

        try:
            # Update state
            metrics.state = ConnectionState.CONNECTING

            # Create connection
            extra_headers = config.headers if config.headers else None
            ssl_context = config.ssl_context

            connection = await websockets.connect(
                config.url,
                extra_headers=extra_headers,
                ssl=ssl_context,
                ping_interval=config.heartbeat_interval,
                ping_timeout=10,
                max_size=None,  # No size limit
                max_queue=None  # No queue limit
            )

            self.connections[connection_name] = connection
            metrics.state = ConnectionState.CONNECTED
            metrics.connected_since = datetime.now()

            # Authenticate if required
            if config.auth_data:
                await self._authenticate(connection_name)

            # Start message handler
            asyncio.create_task(self._handle_messages(connection_name))

            # Start heartbeat if configured
            if config.heartbeat_interval > 0:
                self.heartbeat_tasks[connection_name] = asyncio.create_task(
                    self._heartbeat_loop(connection_name)
                )

            # Send queued messages
            await self._send_queued_messages(connection_name)

            # Notify handler
            if connection_name in self.connection_handlers:
                self.connection_handlers[connection_name](connection_name, metrics.state)

            metrics.state = ConnectionState.ACTIVE
            return True

        except Exception as e:
            self.logger.error(f"Connection failed for {connection_name}: {e}")
            metrics.state = ConnectionState.FAILED
            metrics.error_count += 1

            # Notify handler
            if connection_name in self.connection_handlers:
                self.connection_handlers[connection_name](connection_name, metrics.state)

            return False

    async def _authenticate(self, connection_name: str):
        """Authenticate connection"""
        config = self.connection_configs[connection_name]
        metrics = self.connection_metrics[connection_name]
        connection = self.connections[connection_name]

        try:
            metrics.state = ConnectionState.AUTHENTICATING

            # Send authentication message
            auth_message = json.dumps(config.auth_data)
            await connection.send(auth_message)

            # Wait for authentication response (implementation specific)
            # This is a simplified version - actual implementation depends on API
            await asyncio.sleep(1.0)  # Give time for auth response

            metrics.state = ConnectionState.AUTHENTICATED

        except Exception as e:
            self.logger.error(f"Authentication failed for {connection_name}: {e}")
            metrics.state = ConnectionState.FAILED
            raise

    async def _handle_messages(self, connection_name: str):
        """Handle incoming messages for a connection"""
        connection = self.connections[connection_name]
        metrics = self.connection_metrics[connection_name]

        try:
            async for message in connection:
                try:
                    # Update metrics
                    metrics.messages_received += 1
                    metrics.last_message = datetime.now()

                    # Parse message
                    data = json.loads(message)

                    # Call message handler if registered
                    if connection_name in self.message_handlers:
                        self.message_handlers[connection_name](connection_name, data)

                except json.JSONDecodeError as e:
                    self.logger.warning(f"Invalid JSON from {connection_name}: {e}")
                    metrics.error_count += 1

                except Exception as e:
                    self.logger.error(f"Message handler error for {connection_name}: {e}")
                    metrics.error_count += 1

        except websockets.exceptions.ConnectionClosed:
            self.logger.warning(f"Connection closed: {connection_name}")
        except Exception as e:
            self.logger.error(f"Message handling error for {connection_name}: {e}")
        finally:
            # Mark connection as disconnected
            metrics.state = ConnectionState.DISCONNECTED
            if connection_name in self.connections:
                del self.connections[connection_name]

            # Trigger reconnection
            if self.running:
                asyncio.create_task(self._reconnect(connection_name))

    async def _heartbeat_loop(self, connection_name: str):
        """Heartbeat loop for connection health"""
        config = self.connection_configs[connection_name]

        while self.running and connection_name in self.connections:
            try:
                await asyncio.sleep(config.heartbeat_interval)

                if connection_name in self.connections:
                    connection = self.connections[connection_name]
                    # Send ping (websockets library handles this automatically)
                    await connection.ping()

            except Exception as e:
                self.logger.warning(f"Heartbeat failed for {connection_name}: {e}")
                break

    async def _reconnect(self, connection_name: str):
        """Reconnect with exponential backoff"""
        config = self.connection_configs[connection_name]
        metrics = self.connection_metrics[connection_name]

        if metrics.reconnect_count >= config.reconnect_attempts:
            self.logger.error(f"Maximum reconnect attempts reached for {connection_name}")
            metrics.state = ConnectionState.FAILED
            return

        try:
            metrics.state = ConnectionState.RECONNECTING
            metrics.reconnect_count += 1

            # Exponential backoff
            delay = config.reconnect_delay * (2 ** (metrics.reconnect_count - 1))
            delay = min(delay, 300)  # Cap at 5 minutes

            self.logger.info(f"Reconnecting {connection_name} in {delay}s (attempt {metrics.reconnect_count})")
            await asyncio.sleep(delay)

            if self.running:
                await self._connect(connection_name)

        except Exception as e:
            self.logger.error(f"Reconnection failed for {connection_name}: {e}")

    def _queue_message(self, connection_name: str, message: Dict[str, Any]):
        """Queue message for offline connection"""
        if connection_name not in self.message_queues:
            self.message_queues[connection_name] = []

        queue = self.message_queues[connection_name]

        # Add message to queue
        queue.append({
            "message": message,
            "timestamp": datetime.now()
        })

        # Limit queue size
        if len(queue) > self.max_queue_size:
            queue.pop(0)  # Remove oldest message

    async def _send_queued_messages(self, connection_name: str):
        """Send queued messages when connection is restored"""
        if connection_name not in self.message_queues:
            return

        queue = self.message_queues[connection_name]
        sent_count = 0

        while queue and connection_name in self.connections:
            try:
                queued_item = queue.pop(0)
                message = queued_item["message"]

                await self.send_message(connection_name, message)
                sent_count += 1

            except Exception as e:
                self.logger.error(f"Error sending queued message: {e}")
                break

        if sent_count > 0:
            self.logger.info(f"Sent {sent_count} queued messages for {connection_name}")

    async def _monitor_connections(self):
        """Monitor connection health and trigger reconnections"""
        while self.running:
            try:
                current_time = datetime.now()

                for name, metrics in self.connection_metrics.items():
                    # Check for stale connections
                    if (metrics.last_message and
                        (current_time - metrics.last_message).total_seconds() > 300):  # 5 minutes
                        self.logger.warning(f"No messages received from {name} for 5 minutes")

                    # Check connection state
                    if (metrics.state == ConnectionState.FAILED and
                        name in self.connection_configs):
                        asyncio.create_task(self._reconnect(name))

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                self.logger.error(f"Connection monitoring error: {e}")

    async def _close_connection(self, connection_name: str):
        """Close specific connection"""
        if connection_name in self.connections:
            try:
                connection = self.connections[connection_name]
                await connection.close()
                del self.connections[connection_name]
            except Exception as e:
                self.logger.warning(f"Error closing connection {connection_name}: {e}")

    async def _close_all_connections(self):
        """Close all connections"""
        for connection_name in list(self.connections.keys()):
            await self._close_connection(connection_name)

    def get_connection_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all connections"""
        status = {}

        for name, metrics in self.connection_metrics.items():
            status[name] = {
                "state": metrics.state.value,
                "connected_since": metrics.connected_since.isoformat() if metrics.connected_since else None,
                "last_message": metrics.last_message.isoformat() if metrics.last_message else None,
                "messages_sent": metrics.messages_sent,
                "messages_received": metrics.messages_received,
                "reconnect_count": metrics.reconnect_count,
                "error_count": metrics.error_count,
                "queued_messages": len(self.message_queues.get(name, []))
            }

        return status

    def __del__(self):
        """Cleanup on destruction"""
        if self.running:
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(self.stop())
            except Exception:
                pass