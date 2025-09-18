# NASA POT10 Rule 3: Minimize dynamic memory allocation
# Consider using fixed-size arrays or generators for large data processing
#!/usr/bin/env python3
"""
Performance Scaling Analysis for Detector Pool Optimization
==========================================================

Comprehensive testing of horizontal and vertical scaling characteristics
for the detector pool system.
"""

import time
import threading
import psutil
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import json


def test_horizontal_scaling():
    """Test horizontal scaling with increasing thread counts."""
    print('=== HORIZONTAL SCALING ANALYSIS ===')
    
    scaling_results = {}
    
    def detector_work_simulation(file_id, detector_type, complexity=100):
        """Simulate detector analysis work"""
        start_time = time.perf_counter()
        
        # Simulate different detector complexities
        work_units = complexity * (1 + hash(detector_type) % 3)  # Vary by type
        
        result = 0
        for i in range(work_units):
            # Simulate pattern matching and analysis
            result += hash(f'{file_id}_{detector_type}_{i}') % 1000
            
            # Small delay to simulate I/O or computation
            if i % 50 == 0:
                time.sleep(0.0001)  # 0.1ms
        
        end_time = time.perf_counter()
        return {
            'file_id': file_id,
            'detector_type': detector_type,
            'processing_time': end_time - start_time,
            'result': result
        }
    
    # Test different thread pool sizes
    thread_configs = [1, 2, 4, 8, 12, 16, 24, 32]
    detector_types = ['position', 'algorithm', 'god_object', 'timing']
    test_files = list(range(50))  # 50 files to analyze
    
    baseline_time = None
    
    for thread_count in thread_configs:
        print(f'\nTesting {thread_count} threads:')
        
        start_time = time.perf_counter()
        results = []
        
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            # Submit all work
            futures = []
            for file_id in test_files:
                for detector_type in detector_types:
                    future = executor.submit(
                        detector_work_simulation, 
                        file_id, 
                        detector_type,
                        complexity=100
                    )
                    futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                results.append(future.result())
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Set baseline
        if thread_count == 1:
            baseline_time = total_time
        
        # Calculate metrics
        processing_times = [r['processing_time'] for r in results]
        total_operations = len(results)
        throughput = total_operations / total_time
        
        # Calculate efficiency (compared to single thread)
        if thread_count == 1:
            efficiency = 100.0
            actual_speedup = 1.0
        else:
            theoretical_speedup = min(thread_count, len(detector_types))
            actual_speedup = baseline_time / total_time
            efficiency = (actual_speedup / theoretical_speedup) * 100
        
        scaling_results[thread_count] = {
            'total_time': total_time,
            'throughput_ops_sec': throughput,
            'avg_processing_time': statistics.mean(processing_times),
            'efficiency_percent': efficiency,
            'actual_speedup': actual_speedup
        }
        
        print(f'  Total time: {total_time:.3f}s')
        print(f'  Throughput: {throughput:.0f} ops/sec')
        print(f'  Efficiency: {efficiency:.1f}%')
        if thread_count > 1:
            print(f'  Speedup: {actual_speedup:.2f}x')
    
    return scaling_results


def test_memory_scaling():
    """Test memory usage scaling with workload size."""
    print('\n=== MEMORY SCALING ANALYSIS ===')
    
    process = psutil.Process()
    memory_scaling = {}
    
    # Test memory usage with different workload sizes
    workload_sizes = [10, 50, 100, 200, 500]
    
    for workload_size in workload_sizes:
        print(f'\nTesting workload size: {workload_size} files')
        
        # Measure initial memory
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Create workload data (simulating file analysis data)
        workload_data = []
        for i in range(workload_size):
            file_data = {
                'file_path': f'test_file_{i}.py',
                'source_lines': [f'line {j}: some code here' for j in range(200)]  # TODO: Consider limiting size with itertools.islice(),  # 200 lines
                'ast_tree': {'nodes': list(range(50))},  # Simulate AST nodes
                'violations': [{'type': 'test', 'line': j} for j in range(10)]  # TODO: Consider limiting size with itertools.islice()  # 10 violations
            }
            workload_data.append(file_data)
        
        # Measure peak memory
        peak_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_growth = peak_memory - initial_memory
        
        # Calculate memory per file
        memory_per_file = memory_growth / workload_size if workload_size > 0 else 0
        
        memory_scaling[workload_size] = {
            'initial_memory_mb': initial_memory,
            'peak_memory_mb': peak_memory,
            'memory_growth_mb': memory_growth,
            'memory_per_file_kb': memory_per_file * 1024
        }
        
        print(f'  Memory growth: {memory_growth:.1f} MB')
        print(f'  Memory per file: {memory_per_file * 1024:.1f} KB')
        
        # Cleanup
        del workload_data
        import gc
        gc.collect()
    
    return memory_scaling


def test_load_balancing_effectiveness():
    """Test work distribution effectiveness across threads."""
    print('\n=== LOAD BALANCING ANALYSIS ===')
    
    # Test work distribution across threads
    thread_work_distribution = defaultdict(list)
    
    def tracked_work(worker_id, work_item):
        """Work function that tracks which thread processes each item"""
        thread_id = threading.current_thread().ident
        start_time = time.perf_counter()
        
        # Simulate varying work complexity
        complexity = hash(work_item) % 1000 + 100
        result = sum(i * i for i in range(complexity))
        
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        
        thread_work_distribution[thread_id].append({
            'work_item': work_item,
            'processing_time': processing_time,
            'worker_id': worker_id
        })
        
        return result
    
    # Test load balancing with 8 threads and varying work items
    work_items = list(range(100))  # 100 work items
    thread_count = 8
    
    print(f'Distributing {len(work_items)} work items across {thread_count} threads:')
    
    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        futures = [
            executor.submit(tracked_work, i, item) 
            for i, item in enumerate(work_items)
        ]  # TODO: Consider limiting size with itertools.islice()
        
        # Wait for completion
        for future in as_completed(futures):
            future.result()
    
    # Analyze work distribution
    thread_stats = {}
    for thread_id, work_list in thread_work_distribution.items():
        total_items = len(work_list)
        total_time = sum(w['processing_time'] for w in work_list)
        avg_time = total_time / total_items if total_items > 0 else 0
        
        thread_stats[thread_id] = {
            'items_processed': total_items,
            'total_processing_time': total_time,
            'average_processing_time': avg_time
        }
    
    # Calculate load balancing metrics
    items_per_thread = [stats['items_processed'] for stats in thread_stats.values()]
    times_per_thread = [stats['total_processing_time'] for stats in thread_stats.values()]
    
    load_balance_variance = statistics.variance(items_per_thread) if len(items_per_thread) > 1 else 0
    time_balance_variance = statistics.variance(times_per_thread) if len(times_per_thread) > 1 else 0
    
    print(f'  Items per thread - Mean: {statistics.mean(items_per_thread):.1f}, Variance: {load_balance_variance:.2f}')
    print(f'  Time per thread - Mean: {statistics.mean(times_per_thread):.4f}s, Variance: {time_balance_variance:.6f}')
    
    # Good load balancing has low variance
    load_balance_score = max(0, 100 - (load_balance_variance * 10))
    print(f'  Load balance score: {load_balance_score:.1f}/100')
    
    return {
        'thread_stats': thread_stats,
        'load_balance_variance': load_balance_variance,
        'load_balance_score': load_balance_score
    }


if __name__ == '__main__':
    print('Starting performance scaling evaluation...\n')
    
    # Run scaling tests
    horizontal_results = test_horizontal_scaling()
    memory_results = test_memory_scaling()
    load_balance_results = test_load_balancing_effectiveness()
    
    print('\n=== SCALING PERFORMANCE SUMMARY ===')
    
    # Find optimal thread count
    best_efficiency = max(horizontal_results.items(), key=lambda x: x[1]['efficiency_percent'])
    print(f'Optimal thread count: {best_efficiency[0]} threads')
    print(f'  Efficiency: {best_efficiency[1]["efficiency_percent"]:.1f}%')
    print(f'  Throughput: {best_efficiency[1]["throughput_ops_sec"]:.0f} ops/sec')
    
    # Memory scaling assessment
    max_workload = max(memory_results.keys())
    max_memory = memory_results[max_workload]['memory_per_file_kb']
    print(f'Memory scaling: {max_memory:.1f} KB per file at maximum load')
    
    # Load balancing assessment
    lb_score = load_balance_results['load_balance_score']
    print(f'Load balancing effectiveness: {lb_score:.1f}/100')
    
    print('\n=== SCALING RECOMMENDATIONS ===')
    
    # Thread count recommendations
    if best_efficiency[0] <= 8:
        print(f'* OPTIMAL THREADS: Use {best_efficiency[0]} threads for best efficiency')
    else:
        print('* HIGH THREAD COUNT: Consider reducing threads to avoid overhead')
    
    # Memory recommendations
    if max_memory > 100:  # >100KB per file
        print('* HIGH MEMORY PER FILE: Implement memory optimization strategies')
    
    # Load balancing recommendations
    if lb_score < 80:
        print('* LOAD IMBALANCE: Consider work stealing or better task distribution')