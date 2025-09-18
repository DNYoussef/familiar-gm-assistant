#!/usr/bin/env python3
"""
Performance Monitoring Script
=============================

Real-time performance monitoring script that integrates with the optimization engine
to provide continuous performance tracking, alerting, and automatic optimization triggers.

Features:
- Real-time performance metric collection
- Automated performance regression detection
- Integration with optimization engine for auto-tuning
- Performance dashboard generation
- Alert management with severity levels
- Historical performance tracking

Usage:
  python scripts/performance_monitor.py --project-path . --monitor-duration 3600
  python scripts/performance_monitor.py --dashboard-only --port 8080
  python scripts/performance_monitor.py --regression-check --baseline-file baseline.json

NASA Rules 4, 5, 6, 7: Function limits, assertions, scoping, bounded resources
"""

import asyncio
import argparse
import json
import time
import signal
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
from lib.shared.utilities import get_logger
logger = get_logger(__name__)


class PerformanceMonitoringDashboard:
    """
    Real-time performance monitoring dashboard with web interface.
    
    NASA Rule 4: All methods under 60 lines
    NASA Rule 7: Bounded resource usage
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize performance monitoring dashboard."""
        self.config = config or self._get_default_config()
        self.monitoring_active = False
        self.performance_history: List[Dict[str, Any]] = []
        self.alert_history: List[PerformanceAlert] = []
        self.optimization_engine = None
        self.real_time_monitor = None
        self.cache_profiler = None
        
        # Initialize monitoring components if available
        if PERFORMANCE_MONITORING_AVAILABLE:
            try:
                self.optimization_engine = get_global_optimization_engine()
                self.real_time_monitor = get_global_real_time_monitor()
                self.cache_profiler = get_global_profiler()
                
                # Set up alert callback
                self.real_time_monitor.alert_callback = self._handle_performance_alert
                
            except Exception as e:
                logger.warning(f"Failed to initialize monitoring components: {e}")
        
        # Performance thresholds
        self.performance_thresholds = self.config.get("performance_thresholds", {})
        self.regression_detection_enabled = self.config.get("regression_detection", {}).get("enabled", True)
        
        # Dashboard state
        self.dashboard_data = {
            "status": "initialized",
            "start_time": time.time(),
            "alerts": [],
            "metrics": {},
            "optimizations": []
        }
        
        logger.info("Performance monitoring dashboard initialized")
    
    async def start_monitoring(self, 
                              project_path: str,
                              duration_seconds: Optional[int] = None) -> None:
        """
        Start performance monitoring for specified duration.
        
        NASA Rule 4: Function under 60 lines
        NASA Rule 5: Input validation
        """
        assert project_path, "project_path cannot be empty"
        
        if not PERFORMANCE_MONITORING_AVAILABLE:
            logger.error("Performance monitoring components not available")
            return
        
        logger.info(f"Starting performance monitoring for: {project_path}")\n        self.monitoring_active = True
        self.dashboard_data["status"] = "monitoring"
        
        # Start real-time monitor
        if self.real_time_monitor:
            await self.real_time_monitor.start_monitoring()
        
        # Start cache profiler monitoring
        if self.cache_profiler:
            await self.cache_profiler.start_monitoring(\n                interval_seconds=self.config.get("monitoring_interval", 30.0)
            )
        
        # Monitoring loop
        monitoring_start = time.time()
        last_optimization_check = monitoring_start
        optimization_interval = self.config.get("optimization_interval", 300.0)  # 5 minutes
        
        try:
            while self.monitoring_active:
                current_time = time.time()
                
                # Check if monitoring duration exceeded
                if duration_seconds and (current_time - monitoring_start) > duration_seconds:
                    logger.info("Monitoring duration completed")
                    break
                
                # Collect performance metrics
                await self._collect_performance_metrics()
                
                # Check for performance regressions
                if self.regression_detection_enabled:
                    await self._check_performance_regressions()
                
                # Trigger optimization if needed
                if (current_time - last_optimization_check) > optimization_interval:
                    await self._check_optimization_triggers(project_path)
                    last_optimization_check = current_time
                
                # Update dashboard data
                self._update_dashboard_data()
                
                # Wait for next monitoring cycle
                await asyncio.sleep(self.config.get("monitoring_interval", 30.0))
                
        except KeyboardInterrupt:
            logger.info("Monitoring interrupted by user")
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
        finally:
            await self.stop_monitoring()
    
    async def stop_monitoring(self) -> None:
        """Stop performance monitoring and cleanup resources."""
        logger.info("Stopping performance monitoring")
        self.monitoring_active = False
        self.dashboard_data["status"] = "stopped"
        
        # Stop monitoring components
        if self.real_time_monitor:
            await self.real_time_monitor.stop_monitoring()
        
        if self.cache_profiler:
            await self.cache_profiler.stop_monitoring()
        
        # Generate final report
        final_report = self._generate_final_monitoring_report()
        
        # Save report to file
        report_file = Path("performance_monitoring_report.json")
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        logger.info(f"Final monitoring report saved to: {report_file}")
    
    async def _collect_performance_metrics(self) -> None:
        """Collect performance metrics from all monitoring components."""
        metrics = {
            "timestamp": time.time(),
            "real_time_monitor": {},
            "cache_profiler": {},
            "optimization_engine": {}
        }
        
        # Collect real-time monitor metrics
        if self.real_time_monitor:
            try:
                monitor_report = self.real_time_monitor.get_monitoring_report()
                metrics["real_time_monitor"] = monitor_report
            except Exception as e:
                logger.debug(f"Failed to collect real-time monitor metrics: {e}")
        
        # Collect cache profiler metrics  
        if self.cache_profiler:
            try:
                profiler_summary = self.cache_profiler.get_performance_summary()
                metrics["cache_profiler"] = profiler_summary
            except Exception as e:
                logger.debug(f"Failed to collect cache profiler metrics: {e}")
        
        # Collect optimization engine metrics
        if self.optimization_engine:
            try:
                engine_status = self.optimization_engine.get_optimization_status()
                metrics["optimization_engine"] = engine_status
            except Exception as e:
                logger.debug(f"Failed to collect optimization engine metrics: {e}")
        
        # Store in history with bounded size (NASA Rule 7)
        self.performance_history.append(metrics)
        if len(self.performance_history) > 1000:  # Keep last 1000 entries
            self.performance_history = self.performance_history[-500:]  # Trim to 500
    
    async def _check_performance_regressions(self) -> None:
        """Check for performance regressions using statistical analysis."""
        if len(self.performance_history) < 10:
            return  # Need more data points
        
        # Get recent metrics for regression analysis
        recent_metrics = self.performance_history[-10:]  # Last 10 samples
        baseline_metrics = self.performance_history[-50:-10] if len(self.performance_history) >= 50 else []
        
        if not baseline_metrics:
            return  # Not enough baseline data
        
        # Analyze key performance indicators
        regression_alerts = []
        
        # Check cache hit rate regression
        if self._check_metric_regression("cache_hit_rate", recent_metrics, baseline_metrics):
            regression_alerts.append("Cache hit rate regression detected")
        
        # Check memory usage regression
        if self._check_metric_regression("memory_usage", recent_metrics, baseline_metrics):
            regression_alerts.append("Memory usage regression detected")
        
        # Check processing time regression
        if self._check_metric_regression("processing_time", recent_metrics, baseline_metrics):
            regression_alerts.append("Processing time regression detected")
        
        # Generate alerts for regressions
        for alert_msg in regression_alerts:
            alert = PerformanceAlert(
                alert_id=f"regression_{int(time.time())}",
                timestamp=time.time(),
                severity=AlertSeverity.WARNING,
                bottleneck_type="performance_regression",
                message=alert_msg,
                metrics={"regression_detection": True}
            )
            await self._handle_performance_alert(alert)
    
    def _check_metric_regression(self, metric_name: str, 
                                recent_data: List[Dict], 
                                baseline_data: List[Dict]) -> bool:
        """Check if a specific metric shows regression."""
        try:
            # Extract metric values (simplified extraction)
            recent_values = []
            baseline_values = []
            
            for data in recent_data:
                # Navigate nested structure to find metric
                value = self._extract_metric_value(data, metric_name)
                if value is not None:
                    recent_values.append(value)
            
            for data in baseline_data:
                value = self._extract_metric_value(data, metric_name)
                if value is not None:
                    baseline_values.append(value)
            
            if not recent_values or not baseline_values:
                return False
            
            # Simple statistical comparison
            import statistics
            recent_avg = statistics.mean(recent_values)
            baseline_avg = statistics.mean(baseline_values)
            
            # Check for significant regression (>15% degradation)
            regression_threshold = 0.15
            
            if metric_name == "cache_hit_rate":
                # Lower is worse for hit rate
                regression = (baseline_avg - recent_avg) / baseline_avg
            else:
                # Higher is worse for time/memory metrics
                regression = (recent_avg - baseline_avg) / baseline_avg
            
            return regression > regression_threshold
            
        except Exception as e:
            logger.debug(f"Error checking regression for {metric_name}: {e}")
            return False
    
    def _extract_metric_value(self, data: Dict, metric_name: str) -> Optional[float]:
        """Extract metric value from nested data structure."""
        # Simplified metric extraction - would be more sophisticated in production
        if metric_name == "cache_hit_rate":
            # Look for cache hit rate in profiler data
            profiler_data = data.get("cache_profiler", {})
            overall_metrics = profiler_data.get("overall_metrics", {})
            for cache_name, cache_metrics in overall_metrics.items():
                hit_rate = cache_metrics.get("hit_rate_percent")
                if hit_rate is not None:
                    return hit_rate
        
        elif metric_name == "memory_usage":
            # Look for memory usage
            monitor_data = data.get("real_time_monitor", {})
            current_metrics = monitor_data.get("current_metrics", {})
            return current_metrics.get("memory_usage_mb")
        
        elif metric_name == "processing_time":
            # Look for processing time metrics
            monitor_data = data.get("real_time_monitor", {})
            current_metrics = monitor_data.get("current_metrics", {})
            return current_metrics.get("latency_p95_ms")
        
        return None
    
    async def _check_optimization_triggers(self, project_path: str) -> None:
        """Check if performance optimization should be triggered."""
        if not self.optimization_engine or len(self.performance_history) < 5:
            return
        
        # Get recent performance data
        recent_metrics = self.performance_history[-5:]
        
        # Check optimization triggers
        should_optimize = False
        optimization_reasons = []
        
        # Check cache performance
        cache_hit_rates = []
        for metrics in recent_metrics:
            profiler_data = metrics.get("cache_profiler", {})
            overall_metrics = profiler_data.get("overall_metrics", {})
            for cache_metrics in overall_metrics.values():
                hit_rate = cache_metrics.get("hit_rate_percent", 100)
                cache_hit_rates.append(hit_rate)
        
        if cache_hit_rates:
            avg_hit_rate = sum(cache_hit_rates) / len(cache_hit_rates)
            target_hit_rate = self.performance_thresholds.get("cache_hit_rate_target", 85.0)
            
            if avg_hit_rate < target_hit_rate:
                should_optimize = True
                optimization_reasons.append(f"Cache hit rate below target: {avg_hit_rate:.1f}% < {target_hit_rate}%")
        
        # Check memory usage
        memory_usages = []
        for metrics in recent_metrics:
            monitor_data = metrics.get("real_time_monitor", {})
            current_metrics = monitor_data.get("current_metrics", {})
            memory_mb = current_metrics.get("memory_usage_mb")
            if memory_mb:
                memory_usages.append(memory_mb)
        
        if memory_usages:
            avg_memory = sum(memory_usages) / len(memory_usages)
            memory_threshold = self.performance_thresholds.get("memory_threshold_mb", 150.0)
            
            if avg_memory > memory_threshold:
                should_optimize = True
                optimization_reasons.append(f"Memory usage above threshold: {avg_memory:.1f}MB > {memory_threshold}MB")
        
        # Trigger optimization if needed
        if should_optimize:
            logger.info(f"Triggering automatic optimization. Reasons: {optimization_reasons}")
            try:
                optimization_result = await optimize_analyzer_performance(project_path, 40.0)
                self.dashboard_data["optimizations"].append({
                    "timestamp": time.time(),
                    "trigger_reasons": optimization_reasons,
                    "result": optimization_result
                })
            except Exception as e:
                logger.error(f"Automatic optimization failed: {e}")
    
    def _handle_performance_alert(self, alert: PerformanceAlert) -> None:
        """Handle performance alert from monitoring system."""
        logger.warning(f"Performance Alert: {alert.severity.value} - {alert.message}")
        
        # Store alert in history
        self.alert_history.append(alert)
        
        # Add to dashboard data
        self.dashboard_data["alerts"].append({
            "timestamp": alert.timestamp,
            "severity": alert.severity.value,
            "message": alert.message,
            "metrics": alert.metrics
        })
        
        # Limit alert history size (NASA Rule 7)
        if len(self.alert_history) > 500:
            self.alert_history = self.alert_history[-250:]
        
        if len(self.dashboard_data["alerts"]) > 100:
            self.dashboard_data["alerts"] = self.dashboard_data["alerts"][-50:]
    
    def _update_dashboard_data(self) -> None:
        """Update dashboard data with latest metrics."""
        if not self.performance_history:
            return
        
        latest_metrics = self.performance_history[-1]
        
        self.dashboard_data.update({
            "last_update": time.time(),
            "monitoring_duration": time.time() - self.dashboard_data["start_time"],
            "metrics_collected": len(self.performance_history),
            "alerts_generated": len(self.alert_history),
            "latest_metrics": latest_metrics,
            "status": "monitoring" if self.monitoring_active else "stopped"
        })
    
    def _generate_final_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive final monitoring report."""
        monitoring_duration = time.time() - self.dashboard_data["start_time"]
        
        # Calculate summary statistics
        alert_counts = {}
        for alert in self.alert_history:
            severity = alert.severity.value
            alert_counts[severity] = alert_counts.get(severity, 0) + 1
        
        # Calculate performance trends
        performance_trends = self._calculate_performance_trends()
        
        report = {
            "monitoring_summary": {
                "start_time": datetime.fromtimestamp(self.dashboard_data["start_time"]).isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": monitoring_duration,
                "metrics_collected": len(self.performance_history),
                "alerts_generated": len(self.alert_history),
                "optimizations_triggered": len(self.dashboard_data["optimizations"])
            },
            "alert_summary": {
                "by_severity": alert_counts,
                "total_alerts": len(self.alert_history)
            },
            "performance_trends": performance_trends,
            "optimization_results": self.dashboard_data["optimizations"],
            "configuration": self.config,
            "recommendations": self._generate_monitoring_recommendations()
        }
        
        return report
    
    def _calculate_performance_trends(self) -> Dict[str, Any]:
        """Calculate performance trends over monitoring period."""
        if len(self.performance_history) < 2:
            return {"insufficient_data": True}
        
        first_metrics = self.performance_history[0]
        last_metrics = self.performance_history[-1]
        
        trends = {
            "monitoring_period": {
                "start": first_metrics.get("timestamp"),
                "end": last_metrics.get("timestamp"),
                "data_points": len(self.performance_history)
            }
        }
        
        # Calculate cache hit rate trend
        first_hit_rate = self._extract_metric_value(first_metrics, "cache_hit_rate")
        last_hit_rate = self._extract_metric_value(last_metrics, "cache_hit_rate")
        
        if first_hit_rate and last_hit_rate:
            hit_rate_change = last_hit_rate - first_hit_rate
            trends["cache_hit_rate_change_percent"] = hit_rate_change
            trends["cache_hit_rate_trend"] = "improving" if hit_rate_change > 0 else "declining"
        
        # Calculate memory usage trend
        first_memory = self._extract_metric_value(first_metrics, "memory_usage")
        last_memory = self._extract_metric_value(last_metrics, "memory_usage")
        
        if first_memory and last_memory:
            memory_change = last_memory - first_memory
            trends["memory_usage_change_mb"] = memory_change
            trends["memory_usage_trend"] = "increasing" if memory_change > 0 else "decreasing"
        
        return trends
    
    def _generate_monitoring_recommendations(self) -> List[str]:
        """Generate recommendations based on monitoring results."""
        recommendations = []
        
        # Alert-based recommendations
        if len(self.alert_history) > 10:
            recommendations.append(
                f"High alert frequency detected ({len(self.alert_history)} alerts). "
                "Consider tuning performance thresholds or implementing proactive optimizations."
            )
        
        # Performance trend recommendations
        trends = self._calculate_performance_trends()
        
        if trends.get("cache_hit_rate_trend") == "declining":
            recommendations.append(
                "Cache hit rate is declining. Consider implementing more aggressive cache warming "
                "or increasing cache size limits."
            )
        
        if trends.get("memory_usage_trend") == "increasing":
            recommendations.append(
                "Memory usage is increasing over time. Investigate potential memory leaks "
                "or implement more frequent garbage collection."
            )
        
        # Optimization recommendations
        if len(self.dashboard_data["optimizations"]) == 0:
            recommendations.append(
                "No automatic optimizations were triggered during monitoring. "
                "Consider lowering optimization thresholds for more proactive tuning."
            )
        
        if not recommendations:
            recommendations.append("Performance monitoring completed successfully with no issues detected.")
        
        return recommendations
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default monitoring configuration."""
        return {
            "monitoring_interval": 30.0,
            "optimization_interval": 300.0,
            "performance_thresholds": {
                "cache_hit_rate_target": 85.0,
                "memory_threshold_mb": 150.0,
                "cpu_threshold_percent": 80.0
            },
            "regression_detection": {
                "enabled": True,
                "threshold_percent": 15.0
            },
            "alerting": {
                "email_notifications": False,
                "webhook_url": None
            }
        }


class PerformanceRegressionChecker:
    """
    Performance regression detection and analysis tool.
    
    NASA Rule 4: All methods under 60 lines
    """
    
    def __init__(self, baseline_file: Optional[str] = None):
        """Initialize performance regression checker."""
        self.baseline_file = baseline_file
        self.baseline_data: Optional[Dict[str, Any]] = None
        self.regression_threshold = 15.0  # 15% degradation threshold
        
        if self.baseline_file and path_exists(self.baseline_file):
            self._load_baseline_data()
    
    def _load_baseline_data(self) -> None:
        """Load baseline performance data."""
        try:
            with open(self.baseline_file, 'r') as f:
                self.baseline_data = json.load(f)
            logger.info(f"Loaded baseline data from: {self.baseline_file}")
        except Exception as e:
            logger.error(f"Failed to load baseline data: {e}")
    
    async def check_regression(self, project_path: str, current_metrics_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Check for performance regression against baseline.
        
        NASA Rule 4: Function under 60 lines
        NASA Rule 5: Input validation
        """
        assert project_path, "project_path cannot be empty"
        
        # Get current performance metrics
        if current_metrics_file and path_exists(current_metrics_file):
            with open(current_metrics_file, 'r') as f:
                current_metrics = json.load(f)
        else:
            # Run performance analysis to get current metrics
            current_metrics = await self._collect_current_metrics(project_path)
        
        # Compare with baseline if available
        regression_results = {
            "regression_detected": False,
            "regression_details": [],
            "current_metrics": current_metrics,
            "baseline_available": self.baseline_data is not None
        }
        
        if self.baseline_data:
            regression_analysis = self._analyze_regression(current_metrics, self.baseline_data)
            regression_results.update(regression_analysis)
        
        return regression_results
    
    async def _collect_current_metrics(self, project_path: str) -> Dict[str, Any]:
        """Collect current performance metrics for comparison."""
        if not PERFORMANCE_MONITORING_AVAILABLE:
            return {"error": "Performance monitoring not available"}
        
        try:
            # Run quick performance optimization to get metrics
            optimization_result = await optimize_analyzer_performance(project_path, 25.0)
            
            # Extract key metrics for comparison
            metrics = {
                "timestamp": time.time(),
                "overall_improvement_percent": optimization_result.get("performance_improvements", {}).get("average_improvement_percent", 0),
                "successful_optimizations": optimization_result.get("optimization_summary", {}).get("successful_optimizations", 0),
                "memory_impact_mb": optimization_result.get("resource_impact", {}).get("total_memory_impact_mb", 0),
                "optimization_time_seconds": optimization_result.get("optimization_summary", {}).get("total_optimization_time_seconds", 0)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect current metrics: {e}")
            return {"error": str(e)}
    
    def _analyze_regression(self, current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance regression between current and baseline metrics."""
        regression_details = []
        regression_detected = False
        
        # Compare key metrics
        metrics_to_compare = [
            "overall_improvement_percent",
            "optimization_time_seconds",
            "memory_impact_mb"
        ]
        
        for metric in metrics_to_compare:
            current_value = current.get(metric)
            baseline_value = baseline.get(metric)
            
            if current_value is not None and baseline_value is not None:
                regression_info = self._compare_metric(metric, current_value, baseline_value)
                if regression_info["regression"]:
                    regression_detected = True
                regression_details.append(regression_info)
        
        return {
            "regression_detected": regression_detected,
            "regression_details": regression_details,
            "analysis_timestamp": time.time()
        }
    
    def _compare_metric(self, metric_name: str, current: float, baseline: float) -> Dict[str, Any]:
        """Compare individual metric for regression."""
        # Determine if lower or higher values are better
        lower_is_better = metric_name in ["optimization_time_seconds", "memory_impact_mb"]
        
        if lower_is_better:
            # For metrics where lower is better
            change_percent = ((current - baseline) / baseline) * 100
            regression = change_percent > self.regression_threshold
        else:
            # For metrics where higher is better
            change_percent = ((baseline - current) / baseline) * 100
            regression = change_percent > self.regression_threshold
        
        return {
            "metric": metric_name,
            "current_value": current,
            "baseline_value": baseline,
            "change_percent": change_percent,
            "regression": regression,
            "severity": "high" if abs(change_percent) > 25 else "medium" if abs(change_percent) > self.regression_threshold else "low"
        }
    
    def save_baseline(self, metrics: Dict[str, Any], baseline_file: str) -> None:
        """Save current metrics as baseline for future comparisons."""
        try:
            with open(baseline_file, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            logger.info(f"Baseline saved to: {baseline_file}")
        except Exception as e:
            logger.error(f"Failed to save baseline: {e}")


def load_config(config_file: Optional[str]) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    if not config_file or not path_exists(config_file):
        return {}
    
    try:
        with open(config_file, 'r') as f:
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                if YAML_AVAILABLE:
                    return yaml.safe_load(f)
                else:
                    logger.warning("YAML not available, cannot load YAML config")
                    return {}
            else:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config file {config_file}: {e}")
        return {}


async def run_performance_monitoring(args: argparse.Namespace) -> None:
    """Run performance monitoring based on command line arguments."""
    # Load configuration
    config = load_config(args.config)
    
    # Initialize dashboard
    dashboard = PerformanceMonitoringDashboard(config)
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        dashboard.monitoring_active = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start monitoring
    await dashboard.start_monitoring(args.project_path, args.monitor_duration)


async def run_regression_check(args: argparse.Namespace) -> None:
    """Run performance regression check."""
    checker = PerformanceRegressionChecker(args.baseline_file)
    
    regression_results = await checker.check_regression(
        args.project_path, args.current_metrics_file
    )
    
    print("Performance Regression Check Results:")
    print("=" * 50)
    
    if regression_results["baseline_available"]:
        if regression_results["regression_detected"]:
            print("[FAIL] REGRESSION DETECTED")
            for detail in regression_results["regression_details"]:
                if detail["regression"]:
                    print(f"  - {detail['metric']}: {detail['change_percent']:.1f}% degradation ({detail['severity']} severity)")
        else:
            print("[OK] NO REGRESSION DETECTED")
            print("Performance is stable or improved compared to baseline")
    else:
        print("[WARN]  No baseline available for comparison")
        if args.save_baseline:
            checker.save_baseline(regression_results["current_metrics"], args.save_baseline)
            print(f"Current metrics saved as baseline: {args.save_baseline}")
    
    # Save detailed results
    results_file = "regression_check_results.json"
    with open(results_file, 'w') as f:
        json.dump(regression_results, f, indent=2, default=str)
    print(f"Detailed results saved to: {results_file}")


def main():
    """Main entry point for performance monitoring script."""
    parser = argparse.ArgumentParser(
        description="Performance Monitoring and Regression Detection Tool"
    )
    
    # Common arguments
    parser.add_argument("--project-path", "-p", default=".", 
                       help="Path to project for monitoring")
    parser.add_argument("--config", "-c", 
                       help="Configuration file (YAML or JSON)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Run performance monitoring")
    monitor_parser.add_argument("--duration", "-d", type=int, dest="monitor_duration",
                               help="Monitoring duration in seconds")
    monitor_parser.add_argument("--dashboard-only", action="store_true",
                               help="Run dashboard only without optimization")
    monitor_parser.add_argument("--port", type=int, default=8080,
                               help="Dashboard port number")
    
    # Regression check command
    regression_parser = subparsers.add_parser("check-regression", help="Check for performance regressions")
    regression_parser.add_argument("--baseline-file", "-b",
                                  help="Baseline performance data file")
    regression_parser.add_argument("--current-metrics-file", "-m",
                                  help="Current metrics file (optional)")
    regression_parser.add_argument("--save-baseline", "-s",
                                  help="Save current metrics as baseline")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run appropriate command
    if args.command == "monitor" or not args.command:
        asyncio.run(run_performance_monitoring(args))
    elif args.command == "check-regression":
        asyncio.run(run_regression_check(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()