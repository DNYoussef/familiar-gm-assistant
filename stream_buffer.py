"""
Stream Buffer
High-performance circular buffer for streaming data with backpressure handling
"""

import asyncio
import threading
import time
from collections import deque
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from lib.shared.utilities import get_logger
logger = get_logger(__name__)

        # Circular buffer implementation
        self.buffer: deque = deque(maxlen=self.capacity)
        self.write_index = 0
        self.read_index = 0

        # Threading synchronization
        self.lock = threading.RLock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)

        # Performance tracking
        self.write_count = 0
        self.read_count = 0
        self.overflow_count = 0
        self.underflow_count = 0
        self.last_stats_time = time.time()

        # Batch configuration
        self.batch_size = min(config.processing.batch_size, self.capacity // 4)
        self.flush_interval = config.streaming.flush_interval

        # Event for async operations
        self.data_available = asyncio.Event()

    async def add_data(self, data: StreamData) -> bool:
        """
        Add data to buffer (non-blocking)

        Args:
            data: StreamData to add

        Returns:
            True if added successfully, False if buffer full
        """
        try:
            with self.lock:
                if len(self.buffer) >= self.capacity:
                    # Buffer is full - handle backpressure
                    self.overflow_count += 1
                    self._handle_backpressure()
                    return False

                self.buffer.append(data)
                self.write_count += 1

                # Notify waiting readers
                self.not_empty.notify()

            # Set async event for awaiting coroutines
            self.data_available.set()
            return True

        except Exception as e:
            self.logger.error(f"Error adding data to buffer: {e}")
            return False

    async def get_batch(self, batch_size: Optional[int] = None) -> List[StreamData]:
        """
        Get batch of data from buffer (non-blocking)

        Args:
            batch_size: Size of batch to retrieve

        Returns:
            List of StreamData items
        """
        batch_size = batch_size or self.batch_size
        batch = []

        try:
            with self.lock:
                # Get available data up to batch size
                available = min(len(self.buffer), batch_size)

                for _ in range(available):
                    if self.buffer:
                        batch.append(self.buffer.popleft())
                        self.read_count += 1
                    else:
                        break

                # Notify waiting writers if space available
                if len(self.buffer) < self.capacity:
                    self.not_full.notify()

            # Clear event if buffer is empty
            if not batch:
                self.data_available.clear()

            return batch

        except Exception as e:
            self.logger.error(f"Error getting batch from buffer: {e}")
            return []

    async def wait_for_data(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for data to become available

        Args:
            timeout: Maximum time to wait (seconds)

        Returns:
            True if data is available, False if timeout
        """
        try:
            await asyncio.wait_for(
                self.data_available.wait(),
                timeout=timeout
            )
            return True
        except asyncio.TimeoutError:
            return False

    def get_data_blocking(self, batch_size: Optional[int] = None) -> List[StreamData]:
        """
        Get batch of data (blocking until available)

        Args:
            batch_size: Size of batch to retrieve

        Returns:
            List of StreamData items
        """
        batch_size = batch_size or self.batch_size
        batch = []

        with self.not_empty:
            while len(batch) < batch_size:
                # Wait for data if buffer is empty
                while not self.buffer:
                    self.not_empty.wait(timeout=self.flush_interval)
                    break  # Exit on timeout

                # Get available data
                available = min(len(self.buffer), batch_size - len(batch))
                for _ in range(available):
                    if self.buffer:
                        batch.append(self.buffer.popleft())
                        self.read_count += 1

                # Break if we got some data and timeout occurred
                if batch and not self.buffer:
                    break

            # Notify waiting writers
            self.not_full.notify_all()

        return batch

    def _handle_backpressure(self):
        """Handle buffer overflow situation"""
        # Drop oldest data to make room (configurable strategy)
        if len(self.buffer) >= self.capacity:
            # Remove oldest 10% of data
            drop_count = max(1, self.capacity // 10)
            for _ in range(drop_count):
                if self.buffer:
                    dropped = self.buffer.popleft()
                    self.logger.warning(f"Dropped data due to backpressure: {dropped.symbol}")

    def get_utilization(self) -> float:
        """Get current buffer utilization (0.0 to 1.0)"""
        with self.lock:
            return len(self.buffer) / self.capacity if self.capacity > 0 else 0.0

    def get_stats(self) -> BufferStats:
        """Get buffer performance statistics"""
        current_time = time.time()
        time_delta = current_time - self.last_stats_time

        with self.lock:
            write_rate = self.write_count / time_delta if time_delta > 0 else 0.0
            read_rate = self.read_count / time_delta if time_delta > 0 else 0.0

            stats = BufferStats(
                size=len(self.buffer),
                capacity=self.capacity,
                utilization=self.get_utilization(),
                write_rate=write_rate,
                read_rate=read_rate,
                overflows=self.overflow_count,
                underflows=self.underflow_count
            )

            # Reset counters
            self.write_count = 0
            self.read_count = 0
            self.last_stats_time = current_time

        return stats

    def get_status(self) -> Dict[str, Any]:
        """Get detailed buffer status"""
        stats = self.get_stats()

        return {
            "size": stats.size,
            "capacity": stats.capacity,
            "utilization_percent": stats.utilization * 100,
            "write_rate_per_sec": stats.write_rate,
            "read_rate_per_sec": stats.read_rate,
            "total_overflows": stats.overflows,
            "total_underflows": stats.underflows,
            "batch_size": self.batch_size,
            "flush_interval_ms": self.flush_interval * 1000
        }

    def clear(self):
        """Clear all data from buffer"""
        with self.lock:
            self.buffer.clear()
            self.write_index = 0
            self.read_index = 0

            # Notify all waiting threads
            self.not_full.notify_all()

        # Clear async event
        self.data_available.clear()

    def resize(self, new_capacity: int):
        """Resize buffer capacity"""
        if new_capacity <= 0:
            raise ValueError("Buffer capacity must be positive")

        with self.lock:
            old_capacity = self.capacity
            self.capacity = new_capacity

            # Create new deque with new capacity
            old_buffer = list(self.buffer)
            self.buffer = deque(maxlen=new_capacity)

            # Copy data to new buffer (keep most recent if downsizing)
            if len(old_buffer) > new_capacity:
                # Keep most recent data
                for item in old_buffer[-new_capacity:]:
                    self.buffer.append(item)
                self.overflow_count += len(old_buffer) - new_capacity
            else:
                for item in old_buffer:
                    self.buffer.append(item)

            # Update batch size
            self.batch_size = min(config.processing.batch_size, new_capacity // 4)

            self.logger.info(f"Buffer resized from {old_capacity} to {new_capacity}")

    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        with self.lock:
            return len(self.buffer) == 0

    def is_full(self) -> bool:
        """Check if buffer is full"""
        with self.lock:
            return len(self.buffer) >= self.capacity

    def __len__(self) -> int:
        """Get current buffer size"""
        with self.lock:
            return len(self.buffer)

    def __del__(self):
        """Cleanup buffer resources"""
        try:
            self.clear()
        except Exception:
            pass