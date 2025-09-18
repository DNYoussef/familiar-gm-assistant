"""
Metrics Collector
Advanced metrics collection and aggregation system
"""

import asyncio
from lib.shared.utilities import get_logger
logger = get_logger(__name__)

        # Metric storage
        self.raw_metrics: deque = deque(maxlen=50000)  # Raw metric points
        self.aggregated_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.histogram_data: Dict[str, List[float]] = defaultdict(list)

        # Metric registry
        self.registered_metrics: Dict[str, MetricType] = {}
        self.metric_descriptions: Dict[str, str] = {}

        # Aggregation configuration
        self.aggregation_interval = 60.0  # seconds
        self.histogram_window = 300.0     # 5 minutes

        # Performance tracking
        self.collection_stats = {
            "metrics_collected": 0,
            "aggregations_performed": 0,
            "collection_errors": 0
        }

        # Background tasks
        self.aggregation_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None

        # Built-in metrics
        self._register_builtin_metrics()

        self.running = False

    async def start(self):
        """Start metrics collection system"""
        self.logger.info("Starting metrics collector")
        self.running = True

        # Start background tasks
        self.aggregation_task = asyncio.create_task(self._aggregation_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self):
        """Stop metrics collection system"""
        self.logger.info("Stopping metrics collector")
        self.running = False

        # Cancel tasks
        if self.aggregation_task:
            self.aggregation_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()

    def register_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str = "",
        labels: Optional[List[str]] = None
    ):
        """Register a new metric for collection"""
        self.registered_metrics[name] = metric_type
        if description:
            self.metric_descriptions[name] = description

        self.logger.debug(f"Registered metric: {name} ({metric_type.value})")

    def increment_counter(
        self,
        name: str,
        value: int = 1,
        labels: Optional[Dict[str, str]] = None
    ):
        """Increment a counter metric"""
        self._record_metric(name, value, MetricType.COUNTER, labels)

    def set_gauge(
        self,
        name: str,
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None
    ):
        """Set a gauge metric value"""
        self._record_metric(name, value, MetricType.GAUGE, labels)

    def record_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record a value in a histogram"""
        self._record_metric(name, value, MetricType.HISTOGRAM, labels)

        # Store for histogram calculations
        histogram_key = self._get_metric_key(name, labels)
        self.histogram_data[histogram_key].append(value)

        # Keep only recent values
        cutoff_time = time.time() - self.histogram_window
        self.histogram_data[histogram_key] = [
            v for v in self.histogram_data[histogram_key]
            if v >= cutoff_time  # This is simplified - would need timestamp tracking
        ]

    def time_function(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Decorator/context manager for timing function execution"""
        return TimingContext(self, name, labels)

    def record_processing_time(
        self,
        component: str,
        operation: str,
        duration_ms: float,
        success: bool = True
    ):
        """Record processing time for a component operation"""
        labels = {
            "component": component,
            "operation": operation,
            "status": "success" if success else "error"
        }

        self.record_histogram(f"{component}_processing_time", duration_ms, labels)
        self.increment_counter(f"{component}_operations_total", 1, labels)

    def record_throughput(
        self,
        component: str,
        operation: str,
        items_processed: int,
        duration_seconds: float
    ):
        """Record throughput metrics"""
        rate = items_processed / duration_seconds if duration_seconds > 0 else 0

        labels = {
            "component": component,
            "operation": operation
        }

        self.set_gauge(f"{component}_throughput_rate", rate, labels)
        self.increment_counter(f"{component}_items_processed", items_processed, labels)

    def record_data_quality_metrics(
        self,
        source: str,
        symbol: Optional[str],
        quality_score: float,
        record_count: int,
        error_count: int
    ):
        """Record data quality metrics"""
        labels = {"source": source}
        if symbol:
            labels["symbol"] = symbol

        self.set_gauge("data_quality_score", quality_score, labels)
        self.set_gauge("data_record_count", record_count, labels)
        self.set_gauge("data_error_count", error_count, labels)

        error_rate = error_count / record_count if record_count > 0 else 0
        self.set_gauge("data_error_rate", error_rate, labels)

    def record_api_metrics(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: float,
        request_size_bytes: Optional[int] = None,
        response_size_bytes: Optional[int] = None
    ):
        """Record API-related metrics"""
        labels = {
            "endpoint": endpoint,
            "method": method,
            "status_code": str(status_code),
            "status_class": f"{status_code // 100}xx"
        }

        self.increment_counter("api_requests_total", 1, labels)
        self.record_histogram("api_response_time", response_time_ms, labels)

        if request_size_bytes is not None:
            self.record_histogram("api_request_size_bytes", request_size_bytes, labels)

        if response_size_bytes is not None:
            self.record_histogram("api_response_size_bytes", response_size_bytes, labels)

    def _record_metric(
        self,
        name: str,
        value: Union[int, float],
        metric_type: MetricType,
        labels: Optional[Dict[str, str]] = None
    ):
        """Internal method to record a metric"""
        try:
            metric = Metric(
                name=name,
                value=value,
                timestamp=datetime.now(),
                labels=labels or {},
                metric_type=metric_type
            )

            self.raw_metrics.append(metric)
            self.collection_stats["metrics_collected"] += 1

        except Exception as e:
            self.logger.error(f"Error recording metric {name}: {e}")
            self.collection_stats["collection_errors"] += 1

    def _get_metric_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Generate a unique key for a metric with labels"""
        if not labels:
            return name

        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    async def _aggregation_loop(self):
        """Background task for metric aggregation"""
        while self.running:
            try:
                await self._aggregate_metrics()
                await asyncio.sleep(self.aggregation_interval)
            except Exception as e:
                self.logger.error(f"Metrics aggregation error: {e}")

    async def _aggregate_metrics(self):
        """Aggregate raw metrics into time series"""
        if not self.raw_metrics:
            return

        current_time = datetime.now()
        cutoff_time = current_time - timedelta(seconds=self.aggregation_interval)

        # Get metrics from the aggregation window
        window_metrics = [
            metric for metric in self.raw_metrics
            if metric.timestamp >= cutoff_time
        ]

        if not window_metrics:
            return

        # Group metrics by name and labels
        grouped_metrics = defaultdict(list)
        for metric in window_metrics:
            key = self._get_metric_key(metric.name, metric.labels)
            grouped_metrics[key].append(metric)

        # Aggregate each group
        for metric_key, metrics in grouped_metrics.items():
            await self._aggregate_metric_group(metric_key, metrics, current_time)

        self.collection_stats["aggregations_performed"] += 1

    async def _aggregate_metric_group(
        self,
        metric_key: str,
        metrics: List[Metric],
        timestamp: datetime
    ):
        """Aggregate a group of metrics with the same name and labels"""
        if not metrics:
            return

        metric_type = metrics[0].metric_type
        metric_name = metrics[0].name

        if metric_type == MetricType.COUNTER:
            # Sum all counter increments
            total_value = sum(metric.value for metric in metrics)
            aggregated_value = total_value

        elif metric_type == MetricType.GAUGE:
            # Use the latest gauge value
            aggregated_value = metrics[-1].value

        elif metric_type in [MetricType.HISTOGRAM, MetricType.TIMER]:
            # Calculate histogram statistics
            values = [metric.value for metric in metrics]
            aggregated_value = {
                "count": len(values),
                "sum": sum(values),
                "min": min(values),
                "max": max(values),
                "avg": np.mean(values),
                "p50": np.percentile(values, 50),
                "p95": np.percentile(values, 95),
                "p99": np.percentile(values, 99)
            }

        else:
            aggregated_value = metrics[-1].value

        # Store aggregated metric
        aggregated_metric = {
            "timestamp": timestamp,
            "value": aggregated_value,
            "labels": metrics[0].labels,
            "type": metric_type.value,
            "count": len(metrics)
        }

        self.aggregated_metrics[metric_key].append(aggregated_metric)

    async def _cleanup_loop(self):
        """Background task for cleaning up old metrics"""
        while self.running:
            try:
                await self._cleanup_old_metrics()
                await asyncio.sleep(3600)  # Cleanup hourly
            except Exception as e:
                self.logger.error(f"Metrics cleanup error: {e}")

    async def _cleanup_old_metrics(self):
        """Clean up old metric data"""
        cutoff_time = datetime.now() - timedelta(hours=24)

        # Clean up raw metrics
        self.raw_metrics = deque([
            metric for metric in self.raw_metrics
            if metric.timestamp >= cutoff_time
        ], maxlen=50000)

        # Clean up aggregated metrics
        for key in self.aggregated_metrics:
            self.aggregated_metrics[key] = deque([
                metric for metric in self.aggregated_metrics[key]
                if metric["timestamp"] >= cutoff_time
            ], maxlen=1000)

    def get_metrics_summary(self, component: Optional[str] = None, minutes: int = 60) -> Dict[str, Any]:
        """Get metrics summary for dashboard"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)

        # Filter metrics by component and time
        relevant_metrics = []
        for metric_key, metrics in self.aggregated_metrics.items():
            for metric in metrics:
                if metric["timestamp"] >= cutoff_time:
                    if not component or component in metric.get("labels", {}).get("component", ""):
                        relevant_metrics.append((metric_key, metric))

        if not relevant_metrics:
            return {"error": "No metrics found for specified criteria"}

        # Aggregate by type
        summary = {
            "timeframe_minutes": minutes,
            "component": component or "all",
            "counters": {},
            "gauges": {},
            "histograms": {},
            "timestamp": datetime.now().isoformat()
        }

        for metric_key, metric in relevant_metrics:
            metric_name = metric_key.split('{')[0]  # Remove labels from key

            if metric["type"] == "counter":
                if metric_name not in summary["counters"]:
                    summary["counters"][metric_name] = 0
                summary["counters"][metric_name] += metric["value"]

            elif metric["type"] == "gauge":
                summary["gauges"][metric_name] = metric["value"]

            elif metric["type"] in ["histogram", "timer"]:
                if isinstance(metric["value"], dict):
                    summary["histograms"][metric_name] = metric["value"]

        return summary

    def get_component_metrics(self, component: str, hours: int = 1) -> MetricsSummary:
        """Get comprehensive metrics for a specific component"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # Filter metrics for this component
        component_metrics = []
        for metric_key, metrics in self.aggregated_metrics.items():
            for metric in metrics:
                if (metric["timestamp"] >= cutoff_time and
                    metric.get("labels", {}).get("component") == component):
                    component_metrics.append((metric_key, metric))

        # Aggregate by type
        counters = {}
        gauges = {}
        timers = {}
        throughput = {}

        for metric_key, metric in component_metrics:
            metric_name = metric_key.split('{')[0]

            if metric["type"] == "counter":
                if metric_name not in counters:
                    counters[metric_name] = 0
                counters[metric_name] += metric["value"]

                # Calculate throughput (ops per second)
                if "operations" in metric_name:
                    throughput[metric_name] = metric["value"] / (hours * 3600)

            elif metric["type"] == "gauge":
                gauges[metric_name] = metric["value"]

            elif metric["type"] in ["histogram", "timer"]:
                if isinstance(metric["value"], dict):
                    timers[metric_name] = {
                        "avg": metric["value"].get("avg", 0),
                        "p95": metric["value"].get("p95", 0),
                        "p99": metric["value"].get("p99", 0),
                        "count": metric["value"].get("count", 0)
                    }

        return MetricsSummary(
            component=component,
            timestamp=datetime.now(),
            counters=counters,
            gauges=gauges,
            timers=timers,
            throughput=throughput
        )

    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format"""
        output_lines = []

        # Get recent aggregated metrics
        cutoff_time = datetime.now() - timedelta(minutes=5)

        for metric_key, metrics in self.aggregated_metrics.items():
            if not metrics:
                continue

            latest_metric = metrics[-1]
            if latest_metric["timestamp"] < cutoff_time:
                continue

            metric_name = metric_key.split('{')[0]
            labels = latest_metric.get("labels", {})

            # Add description if available
            if metric_name in self.metric_descriptions:
                output_lines.append(f"# HELP {metric_name} {self.metric_descriptions[metric_name]}")

            # Add type
            metric_type = latest_metric["type"]
            prometheus_type = "gauge"  # Default to gauge
            if metric_type == "counter":
                prometheus_type = "counter"
            elif metric_type in ["histogram", "timer"]:
                prometheus_type = "histogram"

            output_lines.append(f"# TYPE {metric_name} {prometheus_type}")

            # Format labels
            label_str = ""
            if labels:
                label_pairs = [f'{k}="{v}"' for k, v in labels.items()]
                label_str = "{" + ",".join(label_pairs) + "}"

            # Add metric values
            if isinstance(latest_metric["value"], dict):
                # Histogram metrics
                hist_data = latest_metric["value"]
                for suffix, value in hist_data.items():
                    if suffix in ["count", "sum", "avg", "p50", "p95", "p99"]:
                        full_name = f"{metric_name}_{suffix}"
                        output_lines.append(f"{full_name}{label_str} {value}")
            else:
                # Simple metric
                output_lines.append(f"{metric_name}{label_str} {latest_metric['value']}")

            output_lines.append("")  # Empty line between metrics

        return "\n".join(output_lines)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get metrics collection statistics"""
        return {
            **self.collection_stats,
            "raw_metrics_count": len(self.raw_metrics),
            "aggregated_metrics_count": sum(len(metrics) for metrics in self.aggregated_metrics.values()),
            "registered_metrics_count": len(self.registered_metrics),
            "histogram_data_points": sum(len(values) for values in self.histogram_data.values()),
            "last_aggregation": datetime.now().isoformat() if self.running else None
        }

    def _register_builtin_metrics(self):
        """Register built-in metrics"""
        builtin_metrics = [
            ("data_quality_score", MetricType.GAUGE, "Overall data quality score (0-1)"),
            ("data_record_count", MetricType.GAUGE, "Number of records processed"),
            ("data_error_count", MetricType.GAUGE, "Number of data errors detected"),
            ("data_error_rate", MetricType.GAUGE, "Data error rate (0-1)"),
            ("api_requests_total", MetricType.COUNTER, "Total number of API requests"),
            ("api_response_time", MetricType.HISTOGRAM, "API response time in milliseconds"),
            ("processing_time", MetricType.HISTOGRAM, "Processing time in milliseconds"),
            ("operations_total", MetricType.COUNTER, "Total number of operations"),
            ("throughput_rate", MetricType.GAUGE, "Processing throughput rate (items/second)"),
            ("items_processed", MetricType.COUNTER, "Total items processed"),
            ("memory_usage_bytes", MetricType.GAUGE, "Memory usage in bytes"),
            ("cpu_usage_percent", MetricType.GAUGE, "CPU usage percentage"),
        ]

        for name, metric_type, description in builtin_metrics:
            self.register_metric(name, metric_type, description)


class TimingContext:
    """Context manager for timing operations"""

    def __init__(self, collector: MetricsCollector, name: str, labels: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.name = name
        self.labels = labels or {}
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration_ms = (time.time() - self.start_time) * 1000
            success = exc_type is None

            # Add success/error to labels
            labels = {**self.labels, "status": "success" if success else "error"}

            self.collector.record_histogram(self.name, duration_ms, labels)

    async def __aenter__(self):
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration_ms = (time.time() - self.start_time) * 1000
            success = exc_type is None

            labels = {**self.labels, "status": "success" if success else "error"}
            self.collector.record_histogram(self.name, duration_ms, labels)