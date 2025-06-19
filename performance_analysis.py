#!/usr/bin/env python3
"""
Performance Analysis Script for Neural Network Comparison Project

This script analyzes the performance impact of reorganizing tests into a
dedicated tests/ folder and provides comprehensive benchmarks.
"""

import time
import psutil
import sys
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import json

def get_system_info() -> Dict[str, Any]:
    """Get system information for performance context."""
    return {
        'cpu_count': psutil.cpu_count(),
        'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'python_version': sys.version,
        'platform': sys.platform
    }

def measure_import_performance() -> Dict[str, float]:
    """Measure import performance for different modules."""
    import_times = {}
    
    # Test main module imports
    modules_to_test = [
        'granville_nn',
        'net_torch', 
        'optimized_granville_nn',
        'data_loading'
    ]
    
    for module in modules_to_test:
        start_time = time.perf_counter()
        try:
            exec(f"import {module}")
            import_time = time.perf_counter() - start_time
            import_times[module] = import_time
        except ImportError as e:
            import_times[module] = f"Error: {e}"
    
    return import_times

def measure_test_performance() -> Dict[str, Any]:
    """Measure test execution performance."""
    performance_data = {}
    
    # Run pytest with timing
    start_time = time.perf_counter()
    result = subprocess.run([
        sys.executable, '-m', 'pytest', 'tests/', 
        '--durations=0', '-v', '--tb=no'
    ], capture_output=True, text=True, cwd=os.getcwd())
    
    total_time = time.perf_counter() - start_time
    
    performance_data['total_execution_time'] = total_time
    performance_data['exit_code'] = result.returncode
    performance_data['stdout_lines'] = len(result.stdout.split('\n'))
    performance_data['stderr_lines'] = len(result.stderr.split('\n'))
    
    # Parse test results
    if 'passed' in result.stdout:
        # Extract test statistics
        lines = result.stdout.split('\n')
        for line in lines:
            if 'passed' in line and '=' in line:
                performance_data['test_results'] = line.strip()
                break
    
    return performance_data

def measure_memory_usage() -> Dict[str, float]:
    """Measure memory usage during operations."""
    process = psutil.Process()
    
    # Baseline memory
    baseline_memory = process.memory_info().rss
    
    # Import all modules
    start_memory = process.memory_info().rss
    import granville_nn
    import net_torch
    import optimized_granville_nn
    import data_loading
    post_import_memory = process.memory_info().rss
    
    # Create some data
    import numpy as np
    X, y, _ = data_loading.load_and_analyze_dataset('california_housing')
    post_data_memory = process.memory_info().rss
    
    return {
        'baseline_mb': baseline_memory / 1024 / 1024,
        'post_import_mb': post_import_memory / 1024 / 1024,
        'post_data_mb': post_data_memory / 1024 / 1024,
        'import_overhead_mb': (post_import_memory - start_memory) / 1024 / 1024,
        'data_overhead_mb': (post_data_memory - post_import_memory) / 1024 / 1024
    }

def analyze_project_structure() -> Dict[str, Any]:
    """Analyze the project structure and organization."""
    project_root = Path('.')
    
    structure = {
        'total_files': 0,
        'python_files': 0,
        'test_files': 0,
        'notebook_files': 0,
        'documentation_files': 0,
        'folders': [],
        'file_sizes': {}
    }
    
    for item in project_root.rglob('*'):
        if item.is_file():
            structure['total_files'] += 1
            file_size = item.stat().st_size
            structure['file_sizes'][str(item)] = file_size
            
            if item.suffix == '.py':
                structure['python_files'] += 1
                if 'test' in item.name.lower():
                    structure['test_files'] += 1
            elif item.suffix == '.ipynb':
                structure['notebook_files'] += 1
            elif item.suffix in ['.md', '.txt', '.rst']:
                structure['documentation_files'] += 1
        elif item.is_dir() and not item.name.startswith('.'):
            structure['folders'].append(str(item))
    
    return structure

def generate_performance_report() -> Dict[str, Any]:
    """Generate comprehensive performance report."""
    print("🔍 Analyzing Neural Network Comparison Project Performance...")
    print("=" * 60)
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'system_info': get_system_info(),
        'project_structure': analyze_project_structure(),
        'import_performance': measure_import_performance(),
        'memory_usage': measure_memory_usage(),
        'test_performance': measure_test_performance()
    }
    
    return report

def print_performance_summary(report: Dict[str, Any]) -> None:
    """Print a formatted performance summary."""
    print("\n📊 PERFORMANCE ANALYSIS SUMMARY")
    print("=" * 60)
    
    # System Info
    sys_info = report['system_info']
    print(f"🖥️  System: {sys_info['cpu_count']} CPUs, {sys_info['memory_total']/1024/1024/1024:.1f}GB RAM")
    print(f"🐍 Python: {sys_info['python_version'].split()[0]}")
    
    # Project Structure
    structure = report['project_structure']
    print(f"\n📁 Project Structure:")
    print(f"   • Total files: {structure['total_files']}")
    print(f"   • Python files: {structure['python_files']}")
    print(f"   • Test files: {structure['test_files']}")
    print(f"   • Notebooks: {structure['notebook_files']}")
    print(f"   • Documentation: {structure['documentation_files']}")
    print(f"   • Folders: {', '.join(structure['folders'])}")
    
    # Import Performance
    imports = report['import_performance']
    print(f"\n⚡ Import Performance:")
    for module, time_val in imports.items():
        if isinstance(time_val, float):
            print(f"   • {module}: {time_val*1000:.2f}ms")
        else:
            print(f"   • {module}: {time_val}")
    
    # Memory Usage
    memory = report['memory_usage']
    print(f"\n💾 Memory Usage:")
    print(f"   • Baseline: {memory['baseline_mb']:.1f}MB")
    print(f"   • After imports: {memory['post_import_mb']:.1f}MB (+{memory['import_overhead_mb']:.1f}MB)")
    print(f"   • After data loading: {memory['post_data_mb']:.1f}MB (+{memory['data_overhead_mb']:.1f}MB)")
    
    # Test Performance
    test_perf = report['test_performance']
    print(f"\n🧪 Test Performance:")
    print(f"   • Total execution time: {test_perf['total_execution_time']:.2f}s")
    print(f"   • Exit code: {test_perf['exit_code']}")
    if 'test_results' in test_perf:
        print(f"   • Results: {test_perf['test_results']}")
    
    # Performance Assessment
    print(f"\n✅ PERFORMANCE ASSESSMENT:")
    
    # Import speed assessment
    avg_import_time = sum(t for t in imports.values() if isinstance(t, float)) / len([t for t in imports.values() if isinstance(t, float)])
    if avg_import_time < 0.1:
        print("   🚀 Import speed: EXCELLENT (< 100ms average)")
    elif avg_import_time < 0.5:
        print("   ✅ Import speed: GOOD (< 500ms average)")
    else:
        print("   ⚠️  Import speed: SLOW (> 500ms average)")
    
    # Memory efficiency
    if memory['import_overhead_mb'] < 50:
        print("   💾 Memory efficiency: EXCELLENT (< 50MB overhead)")
    elif memory['import_overhead_mb'] < 100:
        print("   ✅ Memory efficiency: GOOD (< 100MB overhead)")
    else:
        print("   ⚠️  Memory efficiency: HIGH USAGE (> 100MB overhead)")
    
    # Test organization
    test_ratio = structure['test_files'] / structure['python_files'] if structure['python_files'] > 0 else 0
    if test_ratio > 0.3:
        print("   🧪 Test coverage organization: EXCELLENT (> 30% test files)")
    elif test_ratio > 0.2:
        print("   ✅ Test coverage organization: GOOD (> 20% test files)")
    else:
        print("   ⚠️  Test coverage organization: NEEDS IMPROVEMENT (< 20% test files)")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    try:
        report = generate_performance_report()
        print_performance_summary(report)
        
        # Save detailed report
        with open('performance_analysis.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("\n📄 Detailed report saved to: performance_analysis.json")
        
    except Exception as e:
        print(f"❌ Error during performance analysis: {e}")
        sys.exit(1)
