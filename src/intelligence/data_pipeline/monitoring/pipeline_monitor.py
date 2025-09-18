"""
Pipeline Monitor
Comprehensive monitoring system for data pipeline performance and health
"""

import asyncio
from lib.shared.utilities import get_logger
logger = get_logger(__name__)

        # Component tracking
        self.components: Dict[str, ComponentStatus] = {}
        self.component_start_times: Dict[str, datetime] = {}

        # Metrics storage
        self.system_metrics: deque = deque(maxlen=1000)
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))

        # Alerting
        self.alerts: deque = deque(maxlen=1000)
        self.alert_rules = self._load_alert_rules()
        self.alert_callbacks: List[Callable] = []

        # Performance thresholds
        self.performance_thresholds = {
            "cpu_usage_critical": 90.0,
            "memory_usage_critical": 85.0,
            "error_rate_warning": 0.05,
            "error_rate_critical": 0.10,
            "response_time_warning": 1000,  # ms
            "response_time_critical": 5000  # ms
        }

        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.metrics_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None

        # Thread safety
        self.lock = threading.RLock()

        self.running = False

    async def start(self):
        """Start pipeline monitoring"""
        self.logger.info("Starting pipeline monitor")
        self.running = True

        # Start background tasks
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.metrics_task = asyncio.create_task(self._metrics_collection_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self):
        """Stop pipeline monitoring"""
        self.logger.info("Stopping pipeline monitor")
        self.running = False

        # Cancel tasks
        for task in [self.monitoring_task, self.metrics_task, self.cleanup_task]:
            if task:
                task.cancel()

    def register_component(self, component_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Register a component for monitoring"""
        with self.lock:
            self.component_start_times[component_name] = datetime.now()
            self.components[component_name] = ComponentStatus(
                name=component_name,
                status="running",
                uptime_seconds=0.0,
                last_activity=datetime.now(),
                processing_rate=0.0,
                error_count=0,
                warning_count=0,
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
                metadata=metadata or {}
            )

        self.logger.info(f"Registered component: {component_name}")

    def unregister_component(self, component_name: str):
        """Unregister a component from monitoring"""
        with self.lock:
            if component_name in self.components:
                del self.components[component_name]
            if component_name in self.component_start_times:
                del self.component_start_times[component_name]
            if component_name in self.performance_history:
                del self.performance_history[component_name]

        self.logger.info(f"Unregistered component: {component_name}")

    def record_activity(
        self,
        component_name: str,
        activity_type: str = "processed_item",
        count: int = 1,
        processing_time_ms: Optional[float] = None,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record component activity for monitoring"""
        with self.lock:
            if component_name not in self.components:
                self.register_component(component_name)

            component = self.components[component_name]
            component.last_activity = datetime.now()

            # Update counters
            if not success:
                component.error_count += 1

            # Record performance data
            if processing_time_ms is not None:
                self.performance_history[f"{component_name}_processing_time"].append({
                    "timestamp": datetime.now(),
                    "value": processing_time_ms,
                    "success": success
                })

            # Calculate processing rate (items per second over last minute)
            recent_activities = [
                item for item in self.performance_history[f"{component_name}_processing_time"]
                if (datetime.now() - item["timestamp"]).total_seconds() <= 60
            ]

            if recent_activities:
                successful_activities = [item for item in recent_activities if item["success"]]
                component.processing_rate = len(successful_activities) / 60.0
            else:
                component.processing_rate = 0.0

    def record_error(
        self,
        component_name: str,
        error_message: str,
        error_type: str = "general",
        severity: str = "error"
    ):
        """Record an error for a component"""
        with self.lock:
            if component_name in self.components:
                self.components[component_name].error_count += 1
                if severity == "warning":
                    self.components[component_name].warning_count += 1

        # Generate alert if error rate is high
        asyncio.create_task(self._check_error_rate_alert(component_name))

        self.logger.error(f"Component {component_name} error [{error_type}]: {error_message}")

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                await self._update_component_status()
                await self._collect_system_metrics()
                await self._check_performance_alerts()
                await asyncio.sleep(10)  # Update every 10 seconds
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")

    async def _update_component_status(self):
        """Update component status information"""
        current_time = datetime.now()

        with self.lock:
            for component_name, component in self.components.items():
                # Update uptime
                if component_name in self.component_start_times:
                    uptime_delta = current_time - self.component_start_times[component_name]
                    component.uptime_seconds = uptime_delta.total_seconds()

                # Check if component is still active
                time_since_activity = (current_time - component.last_activity).total_seconds()

                if time_since_activity > 300:  # No activity for 5 minutes
                    if component.status == "running":
                        component.status = "warning"
                        await self._generate_alert(
                            component_name,
                            "component_inactive",
                            "warning",
                            f"Component {component_name} inactive for {time_since_activity/60:.1f} minutes"
                        )
                elif time_since_activity > 900:  # No activity for 15 minutes
                    if component.status != "error":
                        component.status = "error"
                        await self._generate_alert(
                            component_name,
                            "component_dead",
                            "critical",
                            f"Component {component_name} appears to be dead (inactive for {time_since_activity/60:.1f} minutes)"
                        )

                # Update resource usage (simplified - would use actual process monitoring)
                component.memory_usage_mb = self._estimate_component_memory(component_name)
                component.cpu_usage_percent = self._estimate_component_cpu(component_name)

    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100

            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv
            }

            # Process information
            process_count = len(psutil.pids())

            # Thread count (simplified)
            thread_count = threading.active_count()

            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory_percent,
                memory_available_gb=memory_available_gb,
                disk_usage_percent=disk_percent,
                network_io_bytes=network_io,
                process_count=process_count,
                thread_count=thread_count
            )

            self.system_metrics.append(metrics)

        except Exception as e:
            self.logger.error(f"System metrics collection error: {e}")

    async def _metrics_collection_loop(self):
        """Metrics collection and aggregation loop"""
        while self.running:
            try:
                await self._aggregate_performance_metrics()
                await asyncio.sleep(60)  # Aggregate every minute
            except Exception as e:
                self.logger.error(f"Metrics collection loop error: {e}")

    async def _aggregate_performance_metrics(self):
        """Aggregate performance metrics for reporting"""
        current_time = datetime.now()

        # Aggregate component performance
        for component_name in self.components.keys():
            processing_time_key = f"{component_name}_processing_time"

            if processing_time_key in self.performance_history:
                recent_data = [
                    item for item in self.performance_history[processing_time_key]
                    if (current_time - item["timestamp"]).total_seconds() <= 300  # Last 5 minutes
                ]

                if recent_data:
                    processing_times = [item["value"] for item in recent_data if item["success"]]
                    if processing_times:
                        avg_processing_time = np.mean(processing_times)
                        p95_processing_time = np.percentile(processing_times, 95)

                        # Store aggregated metrics
                        self.performance_history[f"{component_name}_avg_processing_time"].append({
                            "timestamp": current_time,
                            "value": avg_processing_time
                        })

                        self.performance_history[f"{component_name}_p95_processing_time"].append({
                            "timestamp": current_time,
                            "value": p95_processing_time
                        })

    async def _check_performance_alerts(self):
        """Check for performance-related alerts"""
        # System resource alerts
        if self.system_metrics:
            latest_metrics = self.system_metrics[-1]

            # CPU alert
            if latest_metrics.cpu_usage_percent > self.performance_thresholds["cpu_usage_critical"]:
                await self._generate_alert(
                    "system",
                    "high_cpu_usage",
                    "critical",
                    f"CPU usage at {latest_metrics.cpu_usage_percent:.1f}%",
                    metric_value=latest_metrics.cpu_usage_percent,
                    threshold=self.performance_thresholds["cpu_usage_critical"]
                )

            # Memory alert
            if latest_metrics.memory_usage_percent > self.performance_thresholds["memory_usage_critical"]:
                await self._generate_alert(
                    "system",
                    "high_memory_usage",
                    "critical",
                    f"Memory usage at {latest_metrics.memory_usage_percent:.1f}%",
                    metric_value=latest_metrics.memory_usage_percent,
                    threshold=self.performance_thresholds["memory_usage_critical"]
                )

    async def _check_error_rate_alert(self, component_name: str):
        """Check if component error rate exceeds thresholds"""
        if component_name not in self.components:
            return

        component = self.components[component_name]
        recent_activities = len(self.performance_history[f"{component_name}_processing_time"])

        if recent_activities > 0:
            error_rate = component.error_count / recent_activities

            if error_rate > self.performance_thresholds["error_rate_critical"]:
                await self._generate_alert(
                    component_name,
                    "high_error_rate",
                    "critical",
                    f"Error rate at {error_rate:.1%}",
                    metric_value=error_rate,
                    threshold=self.performance_thresholds["error_rate_critical"]
                )
            elif error_rate > self.performance_thresholds["error_rate_warning"]:
                await self._generate_alert(
                    component_name,
                    "elevated_error_rate",
                    "warning",
                    f"Error rate at {error_rate:.1%}",
                    metric_value=error_rate,
                    threshold=self.performance_thresholds["error_rate_warning"]
                )

    async def _generate_alert(
        self,
        component: str,
        alert_type: str,
        severity: str,
        message: str,
        metric_value: Optional[float] = None,
        threshold: Optional[float] = None
    ):
        """Generate and process performance alert"""
        alert = PerformanceAlert(
            id=f"{alert_type}_{component}_{int(time.time())}",
            component=component,
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            metric_value=metric_value,
            threshold=threshold
        )

        self.alerts.append(alert)

        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                await asyncio.get_event_loop().run_in_executor(None, callback, alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")

        self.logger.warning(f"Performance Alert [{severity}] {component}: {message}")

    async def _cleanup_loop(self):
        """Cleanup old monitoring data"""
        while self.running:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # Cleanup hourly
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")

    async def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        cutoff_time = datetime.now() - timedelta(hours=24)

        # Clean up performance history
        for key, history in self.performance_history.items():
            self.performance_history[key] = deque([
                item for item in history
                if item["timestamp"] >= cutoff_time
            ], maxlen=500)

        # Clean up old alerts
        self.alerts = deque([
            alert for alert in self.alerts
            if alert.timestamp >= cutoff_time
        ], maxlen=1000)

    def _estimate_component_memory(self, component_name: str) -> float:
        """Estimate memory usage for component (simplified)"""
        # In a real implementation, this would track actual process memory
        return psutil.virtual_memory().used / (1024**2) / max(len(self.components), 1)

    def _estimate_component_cpu(self, component_name: str) -> float:
        """Estimate CPU usage for component (simplified)"""
        # In a real implementation, this would track actual process CPU
        return psutil.cpu_percent() / max(len(self.components), 1)

    def _load_alert_rules(self) -> Dict[str, Any]:
        """Load alert rules configuration"""
        return {
            "error_rate_warning": 0.05,
            "error_rate_critical": 0.10,
            "response_time_warning": 1000,
            "response_time_critical": 5000,
            "cpu_usage_warning": 80.0,
            "cpu_usage_critical": 90.0,
            "memory_usage_warning": 75.0,
            "memory_usage_critical": 85.0
        }

    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add callback for performance alerts"""
        self.alert_callbacks.append(callback)

    def get_component_status(self, component_name: Optional[str] = None) -> Dict[str, Any]:
        """Get status information for components"""
        with self.lock:
            if component_name:
                if component_name in self.components:
                    component = self.components[component_name]
                    return {
                        "name": component.name,
                        "status": component.status,
                        "uptime_seconds": component.uptime_seconds,
                        "uptime_human": self._format_duration(component.uptime_seconds),
                        "last_activity": component.last_activity.isoformat(),
                        "processing_rate": component.processing_rate,
                        "error_count": component.error_count,
                        "warning_count": component.warning_count,
                        "memory_usage_mb": component.memory_usage_mb,
                        "cpu_usage_percent": component.cpu_usage_percent,
                        "metadata": component.metadata
                    }
                else:
                    return {"error": "Component not found"}
            else:
                return {
                    "components": {
                        name: {
                            "status": comp.status,
                            "uptime_seconds": comp.uptime_seconds,
                            "processing_rate": comp.processing_rate,
                            "error_count": comp.error_count,
                            "last_activity": comp.last_activity.isoformat()
                        }
                        for name, comp in self.components.items()
                    }
                }

    def get_system_metrics(self, minutes: int = 60) -> Dict[str, Any]:
        """Get system metrics for specified time period"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)

        recent_metrics = [
            metrics for metrics in self.system_metrics
            if metrics.timestamp >= cutoff_time
        ]

        if not recent_metrics:
            return {"error": "No metrics available for specified time period"}

        # Calculate averages
        avg_cpu = np.mean([m.cpu_usage_percent for m in recent_metrics])
        avg_memory = np.mean([m.memory_usage_percent for m in recent_metrics])
        avg_disk = np.mean([m.disk_usage_percent for m in recent_metrics])

        latest = recent_metrics[-1]

        return {
            "time_period_minutes": minutes,
            "latest_timestamp": latest.timestamp.isoformat(),
            "current_cpu_percent": latest.cpu_usage_percent,
            "current_memory_percent": latest.memory_usage_percent,
            "current_disk_percent": latest.disk_usage_percent,
            "current_memory_available_gb": latest.memory_available_gb,
            "average_cpu_percent": avg_cpu,
            "average_memory_percent": avg_memory,
            "average_disk_percent": avg_disk,
            "process_count": latest.process_count,
            "thread_count": latest.thread_count,
            "network_io": latest.network_io_bytes
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        with self.lock:
            total_components = len(self.components)
            running_components = len([c for c in self.components.values() if c.status == "running"])
            error_components = len([c for c in self.components.values() if c.status == "error"])
            warning_components = len([c for c in self.components.values() if c.status == "warning"])

            total_errors = sum(c.error_count for c in self.components.values())
            total_warnings = sum(c.warning_count for c in self.components.values())

            recent_alerts = [a for a in self.alerts if (datetime.now() - a.timestamp).total_seconds() < 3600]

        system_health = "healthy"
        if error_components > 0:
            system_health = "degraded"
        elif warning_components > total_components * 0.2:  # More than 20% warnings
            system_health = "warning"

        return {
            "system_health": system_health,
            "total_components": total_components,
            "running_components": running_components,
            "warning_components": warning_components,
            "error_components": error_components,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "recent_alerts": len(recent_alerts),
            "uptime_seconds": time.time() - (time.time() if not hasattr(self, 'start_time') else self.start_time),
            "last_update": datetime.now().isoformat()
        }

    def get_recent_alerts(self, hours: int = 24, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent performance alerts"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        filtered_alerts = [
            alert for alert in self.alerts
            if alert.timestamp >= cutoff_time
        ]

        if severity:
            filtered_alerts = [a for a in filtered_alerts if a.severity == severity]

        return [
            {
                "id": alert.id,
                "component": alert.component,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "metric_value": alert.metric_value,
                "threshold": alert.threshold,
                "metadata": alert.metadata
            }
            for alert in sorted(filtered_alerts, key=lambda x: x.timestamp, reverse=True)
        ]

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable form"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        elif seconds < 86400:
            return f"{seconds/3600:.1f}h"
        else:
            return f"{seconds/86400:.1f}d"

    def __del__(self):
        """Cleanup on destruction"""
        if self.running:
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(self.stop())
            except Exception:
                pass