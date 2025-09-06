"""
Graph Generation Module for C2LoadSim

Generates visualization graphs from simulation data and saves them as files.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
import json


class GraphGenerator:
    """
    Generate graphs from simulation log data and save them as files.
    """
    
    def __init__(self, logs_directory: str = "logs", graphs_directory: str = "graphs", 
                 analysis_results_dir: str = None):
        """
        Initialize the graph generator.
        
        Args:
            logs_directory: Directory containing log files
            graphs_directory: Directory to save generated graphs
            analysis_results_dir: Optional specific analysis results directory to use
        """
        self.logs_dir = Path(logs_directory)
        
        # If analysis_results_dir is provided, use it; otherwise use default graphs_directory
        if analysis_results_dir:
            self.graphs_dir = Path(analysis_results_dir) / "graphs"
        else:
            self.graphs_dir = Path(graphs_directory)
            
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_results_dir = analysis_results_dir
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_simulation_data(self) -> Optional[pd.DataFrame]:
        """
        Load simulation log data from CSV.
        
        Returns:
            DataFrame containing simulation data or None if file doesn't exist
        """
        csv_file = self.logs_dir / "simulation_log.csv"
        if not csv_file.exists():
            return None
        
        try:
            df = pd.read_csv(csv_file)
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            print(f"Error loading simulation data: {e}")
            return None
    
    def load_workers_data(self) -> Optional[pd.DataFrame]:
        """
        Load worker log data from CSV.
        
        Returns:
            DataFrame containing worker data or None if file doesn't exist
        """
        csv_file = self.logs_dir / "workers_log.csv"
        if not csv_file.exists():
            return None
        
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                return None
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            print(f"Error loading worker data: {e}")
            return None
    
    def load_jobs_data(self) -> Optional[pd.DataFrame]:
        """
        Load jobs log data from CSV.
        
        Returns:
            DataFrame containing jobs data or None if file doesn't exist
        """
        csv_file = self.logs_dir / "jobs_log.csv"
        if not csv_file.exists():
            return None
        
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                return None
            # Convert timestamp columns to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
            df['completed_at'] = pd.to_datetime(df['completed_at'], errors='coerce')
            return df
        except Exception as e:
            print(f"Error loading jobs data: {e}")
            return None
    
    def generate_demand_and_wait_times_graph(self, save_path: str = None) -> Dict[str, str]:
        """
        Generate individual graphs showing demand, job arrivals, and performance metrics.
        
        Args:
            save_path: Optional base path to save the graphs (will create multiple files)
            
        Returns:
            Dictionary mapping graph names to file paths
        """
        df = self.load_simulation_data()
        if df is None:
            raise ValueError("No simulation data available")
        
        results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert simulation_time to time periods (e.g., minutes)
        df['time_period'] = df['simulation_time'] / 60
        
        # Calculate job arrival rate (jobs created per time period)
        df['job_arrival_rate'] = df['total_jobs_created'].diff().fillna(0)
        # Smooth the arrival rate for better visualization
        window_size = max(1, len(df) // 20)
        df['job_arrival_rate_smooth'] = df['job_arrival_rate'].rolling(window=window_size, center=True).mean().fillna(df['job_arrival_rate'])
        
        # Graph 1: Queue Length and Job Arrivals
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax_twin = ax.twinx()
        
        # Queue length (left y-axis)
        line1 = ax.plot(df['time_period'], df['jobs_in_queue'], 'b-', linewidth=3, label='Jobs in Queue')
        ax.fill_between(df['time_period'], df['jobs_in_queue'], alpha=0.4, color='blue')
        ax.set_ylabel('Jobs in Queue', fontsize=12, color='blue', fontweight='bold')
        ax.tick_params(axis='y', labelcolor='blue')
        
        # Job arrival rate (right y-axis)
        width = df['time_period'].diff().fillna(1).median()
        bars = ax_twin.bar(df['time_period'], df['job_arrival_rate'], width=width, 
                            alpha=0.6, color='lightcoral', label='Job Arrivals/Period')
        line2 = ax_twin.plot(df['time_period'], df['job_arrival_rate_smooth'], 'red', 
                              linewidth=2, label='Arrival Trend')
        ax_twin.set_ylabel('Job Arrival Rate (jobs/period)', fontsize=12, color='red', fontweight='bold')
        ax_twin.tick_params(axis='y', labelcolor='red')
        
        ax.set_title('📊 Queue Demand vs Job Arrival Pattern', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Legend with statistics
        arrival_stats = f"(Avg: {df['job_arrival_rate'].mean():.1f}, Peak: {df['job_arrival_rate'].max():.0f})"
        lines = line1 + [bars] + line2
        labels = ['Jobs in Queue', f'Job Arrivals/Period {arrival_stats}', 'Arrival Trend']
        ax.legend(lines, labels, loc='upper left', fontsize=10)
        
        plt.tight_layout()
        
        # Save graph 1
        save_path_1 = save_path or self.graphs_dir / f"queue_demand_vs_arrivals_{timestamp}.png"
        plt.savefig(save_path_1, dpi=300, bbox_inches='tight')
        plt.close()
        results['queue_demand_vs_arrivals'] = str(save_path_1)
        
        # Graph 2: Jobs Created vs Completed
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(df['time_period'], df['total_jobs_created'], 'g-', linewidth=3, label='Total Jobs Created')
        ax.plot(df['time_period'], df['jobs_completed'], 'orange', linewidth=3, label='Jobs Completed')
        ax.plot(df['time_period'], df['jobs_failed'], 'red', linewidth=3, label='Jobs Failed')
        ax.set_ylabel('Number of Jobs', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.set_title('Job Creation and Completion Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        
        # Save graph 2
        save_path_2 = save_path or self.graphs_dir / f"job_creation_completion_{timestamp}.png"
        plt.savefig(save_path_2, dpi=300, bbox_inches='tight')
        plt.close()
        results['job_creation_completion'] = str(save_path_2)
        
        # Graph 3: Worker Utilization and Throughput
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax_twin = ax.twinx()
        
        # Worker utilization (left y-axis)
        ax.plot(df['time_period'], df['average_worker_utilization'] * 100, 'purple', linewidth=3, label='Worker Utilization (%)')
        ax.set_ylabel('Worker Utilization (%)', fontsize=12, color='purple', fontweight='bold')
        ax.tick_params(axis='y', labelcolor='purple')
        
        # Throughput (right y-axis)
        ax_twin.plot(df['time_period'], df['throughput'], 'orange', linewidth=3, label='Throughput (jobs/sec)')
        ax_twin.set_ylabel('Throughput (jobs/sec)', fontsize=12, color='orange', fontweight='bold')
        ax_twin.tick_params(axis='y', labelcolor='orange')
        
        ax.set_title('System Performance Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)
        
        plt.tight_layout()
        
        # Save graph 3
        save_path_3 = save_path or self.graphs_dir / f"system_performance_{timestamp}.png"
        plt.savefig(save_path_3, dpi=300, bbox_inches='tight')
        plt.close()
        results['system_performance'] = str(save_path_3)
        
        # Graph 4: Cumulative Job Arrivals with Rate Visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax_twin = ax.twinx()
        
        # Cumulative jobs created (left y-axis)
        ax.plot(df['time_period'], df['total_jobs_created'], 'darkgreen', linewidth=3, label='Cumulative Jobs Created')
        ax.fill_between(df['time_period'], df['total_jobs_created'], alpha=0.2, color='darkgreen')
        ax.set_ylabel('Cumulative Jobs', fontsize=12, color='darkgreen', fontweight='bold')
        ax.tick_params(axis='y', labelcolor='darkgreen')
        
        # Job arrival rate visualization (right y-axis)
        width = df['time_period'].diff().fillna(1).median()
        colors = plt.cm.Reds(np.linspace(0.3, 1.0, len(df)))
        for i, (time, rate) in enumerate(zip(df['time_period'], df['job_arrival_rate'])):
            if rate > 0:
                ax_twin.bar(time, rate, width=width, alpha=0.7, color=colors[i % len(colors)])
        
        ax_twin.plot(df['time_period'], df['job_arrival_rate_smooth'], 'darkred', 
                      linewidth=3, label='Arrival Trend', alpha=0.9)
        ax_twin.set_ylabel('Jobs Arrived This Period', fontsize=12, color='red', fontweight='bold')
        ax_twin.tick_params(axis='y', labelcolor='red')
        
        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.set_title('📈 Job Arrival Impact on System Load', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Enhanced legend
        lines4, labels4 = ax.get_legend_handles_labels()
        ax.legend(lines4 + [plt.Line2D([0], [0], color='darkred', linewidth=2)], 
                  labels4 + ['Arrival Trend'], loc='upper left', fontsize=11)
        
        plt.tight_layout()
        
        # Save graph 4
        save_path_4 = save_path or self.graphs_dir / f"job_arrival_impact_{timestamp}.png"
        plt.savefig(save_path_4, dpi=300, bbox_inches='tight')
        plt.close()
        results['job_arrival_impact'] = str(save_path_4)
        
        return results
    
    def generate_worker_workload_graphs(self, save_path: str = None) -> Dict[str, str]:
        """
        Generate individual graphs showing worker workloads for the whole simulation.
        
        Args:
            save_path: Optional base path to save the graphs
            
        Returns:
            Dictionary mapping graph names to file paths
        """
        sim_df = self.load_simulation_data()
        workers_df = self.load_workers_data()
        
        if sim_df is None:
            raise ValueError("No simulation data available")
        
        results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # If we don't have detailed worker logs, create approximated workload from simulation data
        if workers_df is None or workers_df.empty:
            return self._generate_workload_from_simulation_data(sim_df, save_path, timestamp)
        
        # Create individual graphs for each worker
        workers_df = workers_df.copy()
        unique_workers = workers_df['worker_id'].unique()
        
        if len(unique_workers) == 0:
            raise ValueError("No worker data available")
        
        # Convert timestamp to simulation time in minutes
        workers_df['sim_time_minutes'] = (workers_df['timestamp'] - workers_df['timestamp'].min()).dt.total_seconds() / 60
        
        for idx, worker_id in enumerate(unique_workers):
            worker_data = workers_df[workers_df['worker_id'] == worker_id].copy()
            
            if worker_data.empty:
                continue
            
            # Create individual graph for this worker
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax2 = ax.twinx()
            
            # Plot utilization (left y-axis)
            ax.plot(worker_data['sim_time_minutes'], worker_data['utilization'] * 100, 
                   'b-', linewidth=3, label='Utilization (%)')
            ax.fill_between(worker_data['sim_time_minutes'], worker_data['utilization'] * 100, 
                           alpha=0.3, color='blue')
            
            # Plot completed jobs (right y-axis)
            ax2.plot(worker_data['sim_time_minutes'], worker_data['completed_jobs'], 
                    'g-', linewidth=3, label='Completed Jobs')
            
            # Styling
            ax.set_xlabel('Time (minutes)', fontsize=12)
            ax.set_ylabel('Utilization (%)', fontsize=12, color='blue', fontweight='bold')
            ax2.set_ylabel('Completed Jobs', fontsize=12, color='green', fontweight='bold')
            ax.set_title(f'Worker {idx + 1} Workload Analysis', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='y', labelcolor='blue')
            ax2.tick_params(axis='y', labelcolor='green')
            
            # Add legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)
            
            plt.tight_layout()
            
            # Save individual worker graph
            worker_save_path = save_path or self.graphs_dir / f"worker_{idx + 1}_workload_{timestamp}.png"
            plt.savefig(worker_save_path, dpi=300, bbox_inches='tight')
            plt.close()
            results[f'worker_{idx + 1}_workload'] = str(worker_save_path)
        
        return results
    
    def _generate_workload_from_simulation_data(self, df: pd.DataFrame, save_path: str = None, timestamp: str = None) -> Dict[str, str]:
        """
        Generate individual approximated worker workload graphs from simulation data when detailed worker logs aren't available.
        """
        results = {}
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert to time periods
        df['time_period'] = df['simulation_time'] / 60
        
        # Graph 1: Total Worker Utilization
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(df['time_period'], df['average_worker_utilization'] * 100, 'b-', linewidth=3)
        ax.fill_between(df['time_period'], df['average_worker_utilization'] * 100, alpha=0.3)
        ax.set_ylabel('Average Utilization (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.set_title('Average Worker Utilization Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path_1 = save_path or self.graphs_dir / f"avg_worker_utilization_{timestamp}.png"
        plt.savefig(save_path_1, dpi=300, bbox_inches='tight')
        plt.close()
        results['avg_worker_utilization'] = str(save_path_1)
        
        # Graph 2: Active Workers Count
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(df['time_period'], df['active_workers'], 'g-', linewidth=3)
        ax.fill_between(df['time_period'], df['active_workers'], alpha=0.3)
        ax.set_ylabel('Active Workers', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.set_title('Number of Active Workers Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path_2 = save_path or self.graphs_dir / f"active_workers_count_{timestamp}.png"
        plt.savefig(save_path_2, dpi=300, bbox_inches='tight')
        plt.close()
        results['active_workers_count'] = str(save_path_2)
        
        # Graph 3: Workload Distribution (approximated)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        # Calculate work distribution based on queue length and active workers
        df['work_per_worker'] = df['jobs_in_queue'] / df['total_workers'].fillna(1)
        ax.plot(df['time_period'], df['work_per_worker'], 'r-', linewidth=3)
        ax.fill_between(df['time_period'], df['work_per_worker'], alpha=0.3)
        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.set_ylabel('Avg Jobs per Worker', fontsize=12, fontweight='bold')
        ax.set_title('Average Workload Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path_3 = save_path or self.graphs_dir / f"workload_distribution_{timestamp}.png"
        plt.savefig(save_path_3, dpi=300, bbox_inches='tight')
        plt.close()
        results['workload_distribution'] = str(save_path_3)
        
        return results
    
    def generate_job_analysis_graphs(self, save_path: str = None) -> Dict[str, str]:
        """
        Generate comprehensive individual job analysis graphs.
        
        Args:
            save_path: Optional base path to save the graphs
            
        Returns:
            Dictionary mapping graph names to file paths
        """
        jobs_df = self.load_jobs_data()
        
        if jobs_df is None:
            raise ValueError("No jobs data available")
        
        results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Filter completed jobs for analysis
        completed_jobs = jobs_df[jobs_df['status'] == 'completed'].copy()
        
        if completed_jobs.empty:
            raise ValueError("No completed jobs data available")
        
        # Job Size vs Duration Analysis
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        if 'size' in completed_jobs.columns and 'duration' in completed_jobs.columns:
            scatter = ax.scatter(completed_jobs['size'], completed_jobs['duration'], 
                                c=completed_jobs['processing_time'], cmap='viridis', alpha=0.6, s=50)
            ax.set_xlabel('Job Size', fontsize=12, fontweight='bold')
            ax.set_ylabel('Expected Duration', fontsize=12, fontweight='bold')
            ax.set_title('Job Size vs Duration Analysis', fontsize=14, fontweight='bold')
            plt.colorbar(scatter, ax=ax, label='Processing Time (sec)')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Job Size vs Duration data not available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax.set_title('Job Size vs Duration Analysis', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save the graph
        save_path_1 = save_path or self.graphs_dir / f"job_size_vs_duration_{timestamp}.png"
        plt.savefig(save_path_1, dpi=300, bbox_inches='tight')
        plt.close()
        results['job_size_vs_duration'] = str(save_path_1)
        
        return results
    
    def generate_job_arrival_analysis_graph(self, save_path: str = None) -> Dict[str, str]:
        """
        Generate individual dedicated job arrival analysis graphs with detailed metrics.
        
        Args:
            save_path: Optional base path to save the graphs
            
        Returns:
            Dictionary mapping graph names to file paths
        """
        sim_df = self.load_simulation_data()
        jobs_df = self.load_jobs_data()
        
        if sim_df is None:
            raise ValueError("No simulation data available")
        
        results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert simulation_time to time periods (minutes)
        sim_df['time_period'] = sim_df['simulation_time'] / 60
        
        # Calculate job arrival metrics
        sim_df['job_arrival_rate'] = sim_df['total_jobs_created'].diff().fillna(0)
        sim_df['cumulative_jobs'] = sim_df['total_jobs_created']
        
        # Smooth arrival rate for trend analysis
        window_size = max(1, len(sim_df) // 20)
        sim_df['arrival_rate_smooth'] = sim_df['job_arrival_rate'].rolling(window=window_size, center=True).mean().fillna(sim_df['job_arrival_rate'])
        
        # Graph 1: Job Arrival Rate Over Time
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.bar(sim_df['time_period'], sim_df['job_arrival_rate'], 
                width=sim_df['time_period'].diff().fillna(1).median(), 
                alpha=0.6, color='lightcoral', label='Jobs/Period')
        ax.plot(sim_df['time_period'], sim_df['arrival_rate_smooth'], 
                'red', linewidth=3, label='Smoothed Trend')
        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.set_ylabel('Jobs Arrived per Period', fontsize=12, fontweight='bold')
        ax.set_title('Job Arrival Rate Pattern', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path_1 = save_path or self.graphs_dir / f"job_arrival_rate_pattern_{timestamp}.png"
        plt.savefig(save_path_1, dpi=300, bbox_inches='tight')
        plt.close()
        results['job_arrival_rate_pattern'] = str(save_path_1)
        
        # Graph 2: Cumulative Job Arrivals and Queue Size
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax_twin = ax.twinx()
        
        # Cumulative arrivals (left y-axis)
        ax.plot(sim_df['time_period'], sim_df['cumulative_jobs'], 
                'darkgreen', linewidth=3, label='Cumulative Jobs Created')
        ax.set_ylabel('Cumulative Jobs Created', fontsize=12, color='darkgreen', fontweight='bold')
        ax.tick_params(axis='y', labelcolor='darkgreen')
        
        # Queue size (right y-axis)
        ax_twin.fill_between(sim_df['time_period'], sim_df['jobs_in_queue'], 
                              alpha=0.4, color='blue', label='Jobs in Queue')
        ax_twin.plot(sim_df['time_period'], sim_df['jobs_in_queue'], 
                      'blue', linewidth=2)
        ax_twin.set_ylabel('Jobs in Queue', fontsize=12, color='blue', fontweight='bold')
        ax_twin.tick_params(axis='y', labelcolor='blue')
        
        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.set_title('Job Arrivals vs Queue Buildup', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Combined legend
        lines2, labels2 = ax.get_legend_handles_labels()
        lines2_twin, labels2_twin = ax_twin.get_legend_handles_labels()
        ax.legend(lines2 + lines2_twin, labels2 + labels2_twin, loc='upper left', fontsize=11)
        
        plt.tight_layout()
        
        save_path_2 = save_path or self.graphs_dir / f"arrivals_vs_queue_buildup_{timestamp}.png"
        plt.savefig(save_path_2, dpi=300, bbox_inches='tight')
        plt.close()
        results['arrivals_vs_queue_buildup'] = str(save_path_2)
        
        # Graph 3: Arrival Rate Distribution (Histogram)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        arrival_rates = sim_df['job_arrival_rate'][sim_df['job_arrival_rate'] > 0]
        if len(arrival_rates) > 0:
            ax.hist(arrival_rates, bins=20, alpha=0.7, color='orange', edgecolor='black')
            ax.axvline(arrival_rates.mean(), color='red', linestyle='--', 
                       linewidth=3, label=f'Mean: {arrival_rates.mean():.2f}')
            ax.axvline(arrival_rates.median(), color='blue', linestyle='--', 
                       linewidth=3, label=f'Median: {arrival_rates.median():.2f}')
            ax.set_xlabel('Jobs per Period', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax.set_title('Job Arrival Rate Distribution', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
        else:
            ax.text(0.5, 0.5, 'No job arrival data available', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=14,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax.set_title('Job Arrival Rate Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path_3 = save_path or self.graphs_dir / f"arrival_rate_distribution_{timestamp}.png"
        plt.savefig(save_path_3, dpi=300, bbox_inches='tight')
        plt.close()
        results['arrival_rate_distribution'] = str(save_path_3)
        
        # Graph 4: Arrival Impact on System Performance
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax_twin = ax.twinx()
        
        # Arrival rate (left y-axis)
        ax.plot(sim_df['time_period'], sim_df['arrival_rate_smooth'], 
                'red', linewidth=3, label='Arrival Rate (smooth)')
        ax.set_ylabel('Job Arrival Rate', fontsize=12, color='red', fontweight='bold')
        ax.tick_params(axis='y', labelcolor='red')
        
        # System utilization (right y-axis)
        ax_twin.plot(sim_df['time_period'], sim_df['average_worker_utilization'] * 100, 
                      'purple', linewidth=3, label='Worker Utilization (%)')
        ax_twin.set_ylabel('Worker Utilization (%)', fontsize=12, color='purple', fontweight='bold')
        ax_twin.tick_params(axis='y', labelcolor='purple')
        
        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.set_title('Arrival Rate vs System Utilization', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Combined legend
        lines4, labels4 = ax.get_legend_handles_labels()
        lines4_twin, labels4_twin = ax_twin.get_legend_handles_labels()
        ax.legend(lines4 + lines4_twin, labels4 + labels4_twin, loc='upper left', fontsize=11)
        
        plt.tight_layout()
        
        save_path_4 = save_path or self.graphs_dir / f"arrival_vs_utilization_{timestamp}.png"
        plt.savefig(save_path_4, dpi=300, bbox_inches='tight')
        plt.close()
        results['arrival_vs_utilization'] = str(save_path_4)
        
        return results
    
    def generate_all_graphs(self) -> Dict[str, str]:
        """
        Generate all available individual graphs.
        
        Returns:
            Dictionary mapping graph names to file paths
        """
        all_results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            demand_results = self.generate_demand_and_wait_times_graph()
            all_results.update(demand_results)
            print(f"Generated demand and performance graphs: {list(demand_results.keys())}")
        except Exception as e:
            print(f"Failed to generate demand and performance graphs: {e}")
        
        try:
            arrival_results = self.generate_job_arrival_analysis_graph()
            all_results.update(arrival_results)
            print(f"Generated job arrival analysis graphs: {list(arrival_results.keys())}")
        except Exception as e:
            print(f"Failed to generate job arrival analysis graphs: {e}")
        
        try:
            worker_results = self.generate_worker_workload_graphs()
            all_results.update(worker_results)
            print(f"Generated worker workload graphs: {list(worker_results.keys())}")
        except Exception as e:
            print(f"Failed to generate worker workload graphs: {e}")
        
        try:
            job_results = self.generate_job_analysis_graphs()
            all_results.update(job_results)
            print(f"Generated job analysis graphs: {list(job_results.keys())}")
        except Exception as e:
            print(f"Failed to generate job analysis graphs: {e}")
        
        # Create a summary of generated graphs
        if all_results:
            summary_file = self.graphs_dir / f"graphs_summary_{timestamp}.json"
            with open(summary_file, 'w') as f:
                json.dump({
                    'generated_at': timestamp,
                    'total_graphs': len(all_results),
                    'graphs': all_results,
                    'analysis_results_dir': self.analysis_results_dir
                }, f, indent=2)
            print(f"Generated {len(all_results)} individual graphs. Summary saved to: {summary_file}")
        
        return all_results
    
    @classmethod
    def find_latest_analysis_results_dir(cls, base_path: str = ".") -> Optional[str]:
        """
        Find the latest analysis results directory.
        
        Args:
            base_path: Base path to search for analysis results directories
            
        Returns:
            Path to the latest analysis results directory or None if not found
        """
        base_path = Path(base_path)
        analysis_dirs = list(base_path.glob("analysis_results_*"))
        
        if not analysis_dirs:
            return None
        
        # Sort by modification time, most recent first
        latest_dir = max(analysis_dirs, key=lambda x: x.stat().st_mtime)
        return str(latest_dir)
    
    @classmethod
    def create_for_analysis_results(cls, analysis_results_dir: str, 
                                  logs_directory: str = "logs") -> 'GraphGenerator':
        """
        Create a GraphGenerator instance specifically for an analysis results directory.
        
        Args:
            analysis_results_dir: Path to the analysis results directory
            logs_directory: Directory containing log files
            
        Returns:
            GraphGenerator instance configured for the analysis results directory
        """
        return cls(logs_directory=logs_directory, 
                  graphs_directory="graphs",  # Will be overridden
                  analysis_results_dir=analysis_results_dir)
    
    def get_available_graphs(self) -> List[str]:
        """
        Get list of available graph files.
        
        Returns:
            List of graph file paths
        """
        if not self.graphs_dir.exists():
            return []
        
        graph_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.svg', '*.pdf']:
            graph_files.extend(self.graphs_dir.glob(ext))
        
        return [str(f) for f in sorted(graph_files, key=lambda x: x.stat().st_mtime, reverse=True)]
    
    def clear_old_graphs(self, keep_latest: int = 10):
        """
        Clear old graph files, keeping only the latest ones.
        
        Args:
            keep_latest: Number of latest graphs to keep
        """
        graph_files = self.get_available_graphs()
        
        if len(graph_files) > keep_latest:
            files_to_remove = graph_files[keep_latest:]
            for file_path in files_to_remove:
                try:
                    Path(file_path).unlink()
                    print(f"Removed old graph: {file_path}")
                except Exception as e:
                    print(f"Failed to remove {file_path}: {e}")


if __name__ == "__main__":
    # Example usage
    
    # Option 1: Use default graphs directory
    print("Generating graphs with default directory...")
    generator = GraphGenerator()
    graphs = generator.generate_all_graphs()
    
    print("Generated individual graphs:")
    for name, path in graphs.items():
        print(f"  {name}: {path}")
    
    # Option 2: Use latest analysis results directory
    print("\nLooking for latest analysis results directory...")
    latest_analysis_dir = GraphGenerator.find_latest_analysis_results_dir()
    
    if latest_analysis_dir:
        print(f"Found analysis results directory: {latest_analysis_dir}")
        print("Generating graphs for analysis results...")
        
        analysis_generator = GraphGenerator.create_for_analysis_results(latest_analysis_dir)
        analysis_graphs = analysis_generator.generate_all_graphs()
        
        print("Generated analysis graphs:")
        for name, path in analysis_graphs.items():
            print(f"  {name}: {path}")
    else:
        print("No analysis results directories found.")
    
    # Option 3: Specify a particular analysis results directory
    # specific_dir = "analysis_results_20250903_175659"
    # if Path(specific_dir).exists():
    #     specific_generator = GraphGenerator.create_for_analysis_results(specific_dir)
    #     specific_graphs = specific_generator.generate_all_graphs()
    #     print(f"Generated graphs for {specific_dir}:")
    #     for name, path in specific_graphs.items():
    #         print(f"  {name}: {path}")
