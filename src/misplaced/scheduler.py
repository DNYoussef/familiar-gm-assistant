#!/usr/bin/env python3
"""
Scheduler - Friday 6:00pm automation with real cron integration.

Implements actual scheduling automation to replace theater detection findings.
Provides both cron integration and standalone scheduling for the weekly siphon system.

Key Features:
- Real Friday 6:00pm ET automation
- Cron integration with system scheduler
- Standalone Python scheduler
- Comprehensive logging and monitoring
- Error handling and recovery

Security:
- Environment-based configuration
- Secure log file handling
- Process isolation and management
"""

import os
import sys
from lib.shared.utilities import get_logger
logger = get_logger(__name__)
        logger.setLevel(getattr(logging, self.config.log_level))
        
        # Ensure log directory exists
        log_dir = Path(self.config.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(self.config.log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def install_cron_job(self) -> bool:
        """Install cron job for Friday 6:00pm execution.
        
        This creates a real cron job in the system scheduler to ensure
        the weekly siphon runs automatically.
        
        Returns:
            True if cron job installed successfully
        """
        try:
            # Get current script path
            script_path = str(Path(__file__).absolute())
            python_path = sys.executable
            
            # Create cron command
            # Friday at 6:00pm ET (convert to system time if needed)
            cron_time = self._convert_to_cron_time()
            cron_command = f"{python_path} {script_path} --execute-siphon"
            cron_entry = f"{cron_time} {cron_command}"
            
            self.logger.info(f"Installing cron job: {cron_entry}")
            
            # Get current crontab
            try:
                result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
                current_crontab = result.stdout if result.returncode == 0 else ""
            except FileNotFoundError:
                # crontab command not found (Windows or restricted environment)
                self.logger.warning("crontab command not available, using alternative scheduling")
                return self._install_alternative_scheduler()
            
            # Check if our job already exists
            job_marker = "# Weekly Siphon Automator"
            if job_marker in current_crontab:
                self.logger.info("Cron job already exists, updating...")
                # Remove existing job
                lines = current_crontab.split('\n')
                new_lines = []
                skip_next = False
                for line in lines:
                    if job_marker in line:
                        skip_next = True
                        continue
                    if skip_next and line.strip():
                        skip_next = False
                        continue
                    if not skip_next:
                        new_lines.append(line)
                current_crontab = '\n'.join(new_lines)
            
            # Add our job
            new_crontab = current_crontab.rstrip() + '\n' + job_marker + '\n' + cron_entry + '\n'
            
            # Install new crontab
            process = subprocess.Popen(['crontab', '-'], stdin=subprocess.PIPE, text=True)
            process.communicate(input=new_crontab)
            
            if process.returncode == 0:
                self.logger.info("Cron job installed successfully")
                return True
            else:
                self.logger.error(f"Failed to install cron job: return code {process.returncode}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to install cron job: {e}")
            return self._install_alternative_scheduler()
    
    def _convert_to_cron_time(self) -> str:
        """Convert execution time to cron format."""
        # Parse execution time (e.g., "18:00")
        hour, minute = self.config.execution_time.split(':')
        
        # Friday is day 5 in cron (0=Sunday)
        if self.config.execution_day.lower() == 'friday':
            day_of_week = '5'
        else:
            # Map other days if needed
            day_map = {
                'monday': '1', 'tuesday': '2', 'wednesday': '3',
                'thursday': '4', 'friday': '5', 'saturday': '6', 'sunday': '0'
            }
            day_of_week = day_map.get(self.config.execution_day.lower(), '5')
        
        # Return cron format: minute hour * * day_of_week
        return f"{minute} {hour} * * {day_of_week}"
    
    def _install_alternative_scheduler(self) -> bool:
        """Install alternative scheduler for systems without cron."""
        try:
            # Create a Windows Task Scheduler job or systemd service
            if sys.platform.startswith('win'):
                return self._create_windows_task()
            else:
                return self._create_systemd_service()
        except Exception as e:
            self.logger.error(f"Failed to install alternative scheduler: {e}")
            return False
    
    def _create_windows_task(self) -> bool:
        """Create Windows Task Scheduler job."""
        try:
            script_path = str(Path(__file__).absolute())
            python_path = sys.executable
            
            # Create task using schtasks command
            task_name = "WeeklySiphonAutomator"
            
            # Delete existing task if it exists
            subprocess.run([
                'schtasks', '/delete', '/tn', task_name, '/f'
            ], capture_output=True)
            
            # Create new task
            # Friday at 6:00pm weekly
            cmd = [
                'schtasks', '/create',
                '/tn', task_name,
                '/tr', f'"{python_path}" "{script_path}" --execute-siphon',
                '/sc', 'weekly',
                '/d', 'FRI',
                '/st', self.config.execution_time,
                '/f'  # Force creation
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("Windows scheduled task created successfully")
                return True
            else:
                self.logger.error(f"Failed to create Windows task: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Windows task creation failed: {e}")
            return False
    
    def _create_systemd_service(self) -> bool:
        """Create systemd service and timer."""
        # This would create systemd service files for Linux systems
        # For now, log that it's not implemented
        self.logger.warning("Systemd service creation not yet implemented")
        return False
    
    def remove_scheduled_job(self) -> bool:
        """Remove scheduled job from system scheduler.
        
        Returns:
            True if job removed successfully
        """
        try:
            if sys.platform.startswith('win'):
                # Remove Windows task
                result = subprocess.run([
                    'schtasks', '/delete', '/tn', 'WeeklySiphonAutomator', '/f'
                ], capture_output=True)
                return result.returncode == 0
            else:
                # Remove from cron
                result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
                if result.returncode != 0:
                    return True  # No crontab to remove from
                
                current_crontab = result.stdout
                job_marker = "# Weekly Siphon Automator"
                
                if job_marker not in current_crontab:
                    return True  # Job not found, nothing to remove
                
                # Remove our job
                lines = current_crontab.split('\n')
                new_lines = []
                skip_next = False
                for line in lines:
                    if job_marker in line:
                        skip_next = True
                        continue
                    if skip_next and line.strip():
                        skip_next = False
                        continue
                    if not skip_next:
                        new_lines.append(line)
                
                new_crontab = '\n'.join(new_lines)
                
                # Install updated crontab
                process = subprocess.Popen(['crontab', '-'], stdin=subprocess.PIPE, text=True)
                process.communicate(input=new_crontab)
                
                return process.returncode == 0
                
        except Exception as e:
            self.logger.error(f"Failed to remove scheduled job: {e}")
            return False
    
    def execute_scheduled_siphon(self) -> Dict[str, Any]:
        """Execute the scheduled siphon operation.
        
        This is the main method called by the scheduler to run the
        weekly siphon automation.
        
        Returns:
            Dictionary with execution results
        """
        execution_id = f"scheduled_{datetime.now(timezone.utc).isoformat()}"
        
        try:
            self.logger.info(f"Starting scheduled siphon execution: {execution_id}")
            
            # Pre-execution checks
            pre_check_results = self._perform_pre_execution_checks()
            if not pre_check_results['passed']:
                self.logger.error(f"Pre-execution checks failed: {pre_check_results['issues']}")
                return {
                    'execution_id': execution_id,
                    'status': 'failed',
                    'reason': 'pre_execution_checks_failed',
                    'details': pre_check_results,
                    'timestamp': datetime.now(timezone.utc)
                }
            
            # Execute the siphon with retry logic
            for attempt in range(self.config.max_retries + 1):
                try:
                    self.logger.info(f"Siphon execution attempt {attempt + 1}/{self.config.max_retries + 1}")
                    
                    # Execute the weekly siphon
                    siphon_results = self.automator.execute_weekly_siphon()
                    
                    if siphon_results['status'] == 'completed':
                        # Success!
                        execution_results = {
                            'execution_id': execution_id,
                            'status': 'completed',
                            'siphon_results': siphon_results,
                            'attempt': attempt + 1,
                            'timestamp': datetime.now(timezone.utc)
                        }
                        
                        # Record successful execution
                        self.last_execution = datetime.now(timezone.utc)
                        self.execution_history.append(execution_results)
                        self.error_count = 0  # Reset error count on success
                        
                        # Send success notification
                        if self.config.enable_notifications:
                            self._send_notification("Weekly siphon executed successfully", execution_results)
                        
                        self.logger.info(f"Scheduled siphon completed successfully: {execution_id}")
                        return execution_results
                    
                    else:
                        raise Exception(f"Siphon execution returned status: {siphon_results['status']}")
                        
                except Exception as e:
                    self.logger.error(f"Siphon execution attempt {attempt + 1} failed: {e}")
                    
                    if attempt < self.config.max_retries:
                        self.logger.info(f"Retrying in {self.config.retry_delay} seconds...")
                        time.sleep(self.config.retry_delay)
                    else:
                        # All retries exhausted
                        self.error_count += 1
                        error_results = {
                            'execution_id': execution_id,
                            'status': 'failed',
                            'error': str(e),
                            'attempts': self.config.max_retries + 1,
                            'timestamp': datetime.now(timezone.utc)
                        }
                        
                        self.execution_history.append(error_results)
                        
                        # Send error notification
                        if self.config.enable_notifications:
                            self._send_notification("Weekly siphon execution failed", error_results)
                        
                        return error_results
            
        except Exception as e:
            self.logger.error(f"Scheduled siphon execution failed critically: {e}")
            return {
                'execution_id': execution_id,
                'status': 'critical_failure',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc)
            }
    
    def _perform_pre_execution_checks(self) -> Dict[str, Any]:
        """Perform pre-execution health checks."""
        checks = {
            'passed': True,
            'issues': []
        }
        
        try:
            # Check if it's actually Friday (or configured day)
            now = datetime.now(timezone.utc)
            current_day = now.strftime('%A').lower()
            
            if current_day != self.config.execution_day.lower():
                checks['issues'].append(f"Execution day mismatch: expected {self.config.execution_day}, got {current_day}")
                checks['passed'] = False
            
            # Check automator status
            automator_status = self.automator.get_status()
            if not automator_status['automation_enabled']:
                checks['issues'].append("Automation is disabled")
                checks['passed'] = False
            
            # Check for recent execution (avoid double execution)
            if self.last_execution:
                time_since_last = (now - self.last_execution).total_seconds()
                if time_since_last < 3600:  # Less than 1 hour
                    checks['issues'].append(f"Recent execution detected {time_since_last/60:.1f} minutes ago")
                    checks['passed'] = False
            
            # Check error rate
            if self.error_count >= 3:
                checks['issues'].append(f"High error count: {self.error_count}")
                # Don't fail completely, but warn
            
        except Exception as e:
            checks['issues'].append(f"Pre-execution check error: {e}")
            checks['passed'] = False
        
        return checks
    
    def _send_notification(self, message: str, data: Dict[str, Any]) -> None:
        """Send notification about execution status."""
        try:
            if self.config.notification_webhook:
                # Send webhook notification
                import requests
                
                payload = {
                    'text': message,
                    'data': data,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                response = requests.post(self.config.notification_webhook, json=payload, timeout=10)
                if response.status_code == 200:
                    self.logger.info("Notification sent successfully")
                else:
                    self.logger.warning(f"Notification failed: {response.status_code}")
            else:
                self.logger.info(f"Notification (no webhook): {message}")
                
        except Exception as e:
            self.logger.error(f"Failed to send notification: {e}")
    
    def run_standalone_scheduler(self) -> None:
        """Run standalone Python-based scheduler.
        
        This provides an alternative to cron for systems where
        cron is not available or for development/testing.
        """
        self.logger.info("Starting standalone scheduler")
        self.is_running = True
        
        try:
            while self.is_running:
                # Check if it's time to execute
                if self._is_execution_time():
                    self.logger.info("Execution time reached, starting siphon")
                    self.execute_scheduled_siphon()
                    
                    # Sleep until next week to avoid re-execution
                    self._sleep_until_next_week()
                
                # Health check and monitoring
                if self.config.enable_health_monitoring:
                    self._perform_health_check()
                
                # Sleep for health check interval
                time.sleep(self.config.health_check_interval)
                
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, shutting down scheduler")
        except Exception as e:
            self.logger.error(f"Scheduler error: {e}")
        finally:
            self.is_running = False
            self.logger.info("Standalone scheduler stopped")
    
    def _is_execution_time(self) -> bool:
        """Check if current time matches execution schedule."""
        now = datetime.now(timezone.utc)
        
        # Convert to configured timezone (ET)
        if self.config.timezone == 'US/Eastern':
            # Simple EST/EDT handling
            offset_hours = -5 if self._is_standard_time(now) else -4
            local_time = now + timedelta(hours=offset_hours)
        else:
            local_time = now  # Use UTC if timezone not recognized
        
        # Check day of week
        current_day = local_time.strftime('%A').lower()
        if current_day != self.config.execution_day.lower():
            return False
        
        # Check time (within 1 minute window)
        target_hour, target_minute = map(int, self.config.execution_time.split(':'))
        current_hour = local_time.hour
        current_minute = local_time.minute
        
        time_match = (
            current_hour == target_hour and
            abs(current_minute - target_minute) <= 1
        )
        
        return time_match
    
    def _is_standard_time(self, dt: datetime) -> bool:
        """Check if date is in standard time (EST) vs daylight time (EDT)."""
        # Simple approximation: standard time is Nov-Mar
        month = dt.month
        return month in [11, 12, 1, 2, 3]
    
    def _sleep_until_next_week(self) -> None:
        """Sleep until next week's execution time."""
        # Calculate next Friday
        now = datetime.now(timezone.utc)
        days_ahead = 4 - now.weekday()  # Friday is 4 (Monday is 0)
        if days_ahead <= 0:  # Today is Friday or later
            days_ahead += 7
        
        next_friday = now + timedelta(days=days_ahead)
        target_hour, target_minute = map(int, self.config.execution_time.split(':'))
        
        # Set to execution time on next Friday
        next_execution = next_friday.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
        
        # Adjust for timezone (simple EST approximation)
        next_execution += timedelta(hours=5)  # Convert ET to UTC
        
        sleep_seconds = (next_execution - now).total_seconds()
        if sleep_seconds > 0:
            self.logger.info(f"Sleeping until next execution: {next_execution} ({sleep_seconds/3600:.1f} hours)")
            time.sleep(min(sleep_seconds, 3600))  # Sleep max 1 hour at a time for health checks
    
    def _perform_health_check(self) -> None:
        """Perform periodic health checks."""
        try:
            # Check automator status
            status = self.automator.get_status()
            
            # Log key metrics
            self.logger.debug(f"Health check - Automation enabled: {status['automation_enabled']}, "
                            f"Executions: {status['execution_count']}, Errors: {status['error_count']}")
            
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down gracefully")
        self.is_running = False
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        return {
            'is_running': self.is_running,
            'last_execution': self.last_execution,
            'execution_count': len(self.execution_history),
            'error_count': self.error_count,
            'configuration': {
                'execution_day': self.config.execution_day,
                'execution_time': self.config.execution_time,
                'timezone': self.config.timezone
            },
            'recent_executions': self.execution_history[-5:] if self.execution_history else []
        }

def main():
    """Main entry point for scheduler."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Weekly Siphon Scheduler")
    parser.add_argument('--install-cron', action='store_true', help='Install cron job')
    parser.add_argument('--remove-cron', action='store_true', help='Remove cron job')
    parser.add_argument('--execute-siphon', action='store_true', help='Execute siphon (called by scheduler)')
    parser.add_argument('--run-scheduler', action='store_true', help='Run standalone scheduler')
    parser.add_argument('--status', action='store_true', help='Show scheduler status')
    
    args = parser.parse_args()
    
    # Initialize scheduler
    scheduler = WeeklyScheduler()
    
    if args.install_cron:
        print("Installing scheduled job...")
        success = scheduler.install_cron_job()
        print(f"Installation {'successful' if success else 'failed'}")
        sys.exit(0 if success else 1)
    
    elif args.remove_cron:
        print("Removing scheduled job...")
        success = scheduler.remove_scheduled_job()
        print(f"Removal {'successful' if success else 'failed'}")
        sys.exit(0 if success else 1)
    
    elif args.execute_siphon:
        print("Executing scheduled siphon...")
        results = scheduler.execute_scheduled_siphon()
        print(f"Execution completed: {results['status']}")
        if results.get('siphon_results', {}).get('siphon_execution', {}).get('amount_siphoned', 0) > 0:
            print(f"Amount siphoned: ${results['siphon_results']['siphon_execution']['amount_siphoned']:.2f}")
        sys.exit(0 if results['status'] == 'completed' else 1)
    
    elif args.run_scheduler:
        print("Starting standalone scheduler...")
        scheduler.run_standalone_scheduler()
    
    elif args.status:
        status = scheduler.get_scheduler_status()
        print(json.dumps(status, indent=2, default=str))
    
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python scheduler.py --install-cron    # Install Friday 6pm automation")
        print("  python scheduler.py --run-scheduler   # Run standalone scheduler")
        print("  python scheduler.py --execute-siphon  # Manual execution")
        print("  python scheduler.py --status          # Show status")

if __name__ == '__main__':
    main()

# Export for import validation
__all__ = ['WeeklyScheduler', 'SchedulerConfig']