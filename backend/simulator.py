"""
C2LoadSim Backend Simulator

Core Python module for job generation, queue management, and worker emulation
for computation workload simulation.
"""

import uuid
import time
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import deque
import heapq
import threading
import random
from abc import ABC, abstractmethod
import math
import numpy as np
from scipy import stats
import json
import csv
import os
import shutil
from pathlib import Path


# Custom JSON encoder to handle Enum serialization
class SimulationJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for simulation objects."""
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class JobType(Enum):
    """Enumeration of different job types for simulation."""
    MILP = "milp"
    HEURISTIC = "heuristic"
    ML = "ml"
    MIXED = "mixed"


class JobStatus(Enum):
    """Enumeration of job processing states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkerStatus(Enum):
    """Enumeration of worker states."""
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"


@dataclass
class Job:
    """
    Represents an individual job with attributes like size, type, and duration.
    
    Attributes:
        job_id: Unique identifier for the job
        job_type: Type of computation job (MILP, heuristic, ML, mixed)
        size: Size/complexity of the job (affects processing time)
        duration: Expected processing duration in seconds
        priority: Job priority (higher number = higher priority)
        status: Current status of the job
        created_at: Timestamp when job was created
        started_at: Timestamp when job processing started
        completed_at: Timestamp when job processing completed
        metadata: Additional job-specific metadata
    """
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    job_type: JobType = JobType.MIXED
    size: float = 1.0
    duration: float = 1.0
    priority: int = 0
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def start_processing(self) -> None:
        """Mark the job as started and record the start time."""
        self.status = JobStatus.RUNNING
        self.started_at = datetime.now()
    
    def complete_processing(self) -> None:
        """Mark the job as completed and record the completion time."""
        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.now()
    
    def fail_processing(self, error_message: str = "") -> None:
        """Mark the job as failed and record the failure time."""
        self.status = JobStatus.FAILED
        self.completed_at = datetime.now()
        self.metadata["error"] = error_message
    
    def get_processing_time(self) -> Optional[float]:
        """Get the actual processing time if job is completed."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for serialization."""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type.value if hasattr(self.job_type, 'value') else str(self.job_type),
            "size": self.size,
            "duration": self.duration,
            "priority": self.priority,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
            "processing_time": self.get_processing_time()
        }


class QueueManager:
    """
    Manages job queues and worker assignments for the simulation.
    
    Supports different queue strategies (FIFO, priority-based) and handles
    job distribution to available workers.
    """
    
    def __init__(self, max_queue_size: int = 1000, use_priority_queue: bool = False):
        """
        Initialize the queue manager.
        
        Args:
            max_queue_size: Maximum number of jobs that can be queued
            use_priority_queue: Whether to use priority-based scheduling
        """
        self.max_queue_size = max_queue_size
        self.use_priority_queue = use_priority_queue
        self._lock = threading.Lock()
        
        if use_priority_queue:
            # Priority queue (min-heap, so we use negative priority for max-heap behavior)
            self._queue: List[tuple] = []
        else:
            # FIFO queue
            self._queue: deque = deque()
        
        self.completed_jobs: List[Job] = []
        self.failed_jobs: List[Job] = []
        self.total_jobs_processed = 0
        
    def add_job(self, job: Job) -> bool:
        """
        Add a job to the queue.
        
        Args:
            job: Job to add to the queue
            
        Returns:
            True if job was added successfully, False if queue is full
        """
        with self._lock:
            if len(self._queue) >= self.max_queue_size:
                return False
            
            if self.use_priority_queue:
                # Use negative priority for max-heap behavior
                heapq.heappush(self._queue, (-job.priority, job.created_at, job))
            else:
                self._queue.append(job)
            
            return True
    
    def get_next_job(self) -> Optional[Job]:
        """
        Get the next job from the queue based on the scheduling strategy.
        
        Returns:
            Next job to process, or None if queue is empty
        """
        with self._lock:
            if not self._queue:
                return None
            
            if self.use_priority_queue:
                _, _, job = heapq.heappop(self._queue)
            else:
                job = self._queue.popleft()
            
            return job
    
    def peek_next_job(self) -> Optional[Job]:
        """
        Peek at the next job without removing it from the queue.
        
        Returns:
            Next job to be processed, or None if queue is empty
        """
        with self._lock:
            if not self._queue:
                return None
            
            if self.use_priority_queue:
                _, _, job = self._queue[0]
            else:
                job = self._queue[0]
            
            return job
    
    def get_queue_length(self) -> int:
        """Get the current number of jobs in the queue."""
        with self._lock:
            return len(self._queue)
    
    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return self.get_queue_length() == 0
    
    def is_full(self) -> bool:
        """Check if the queue is full."""
        return self.get_queue_length() >= self.max_queue_size
    
    def complete_job(self, job: Job) -> None:
        """
        Mark a job as completed and add it to completed jobs list.
        
        Args:
            job: The completed job
        """
        with self._lock:
            job.complete_processing()
            self.completed_jobs.append(job)
            self.total_jobs_processed += 1
    
    def fail_job(self, job: Job, error_message: str = "") -> None:
        """
        Mark a job as failed and add it to failed jobs list.
        
        Args:
            job: The failed job
            error_message: Description of the failure
        """
        with self._lock:
            job.fail_processing(error_message)
            self.failed_jobs.append(job)
            self.total_jobs_processed += 1
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the queue and job processing.
        
        Returns:
            Dictionary containing queue statistics
        """
        with self._lock:
            return {
                "queue_length": len(self._queue),
                "max_queue_size": self.max_queue_size,
                "queue_utilization": len(self._queue) / self.max_queue_size,
                "completed_jobs": len(self.completed_jobs),
                "failed_jobs": len(self.failed_jobs),
                "total_jobs_processed": self.total_jobs_processed,
                "success_rate": (
                    len(self.completed_jobs) / self.total_jobs_processed 
                    if self.total_jobs_processed > 0 else 0
                ),
                "use_priority_queue": self.use_priority_queue
            }
    
    def clear_completed_jobs(self) -> List[Job]:
        """
        Clear and return the list of completed jobs.
        
        Returns:
            List of completed jobs that were cleared
        """
        with self._lock:
            completed = self.completed_jobs.copy()
            self.completed_jobs.clear()
            return completed
    
    def clear_failed_jobs(self) -> List[Job]:
        """
        Clear and return the list of failed jobs.
        
        Returns:
            List of failed jobs that were cleared
        """
        with self._lock:
            failed = self.failed_jobs.copy()
            self.failed_jobs.clear()
            return failed


class Worker:
    """
    Emulates a computational worker that processes jobs with statistical parameters.
    
    Workers can have different processing capabilities, failure rates, and 
    performance characteristics to simulate realistic computational environments.
    """
    
    def __init__(
        self, 
        worker_id: str = None,
        processing_speed: float = 1.0,
        failure_rate: float = 0.01,
        efficiency_variance: float = 0.1,
        max_concurrent_jobs: int = 1
    ):
        """
        Initialize a worker with statistical parameters.
        
        Args:
            worker_id: Unique identifier for the worker
            processing_speed: Speed multiplier for job processing (1.0 = normal speed)
            failure_rate: Probability of job failure (0.0 to 1.0)
            efficiency_variance: Variance in processing efficiency (affects actual processing time)
            max_concurrent_jobs: Maximum number of jobs this worker can handle simultaneously
        """
        self.worker_id = worker_id or str(uuid.uuid4())
        self.processing_speed = max(0.1, processing_speed)  # Minimum speed
        self.failure_rate = max(0.0, min(1.0, failure_rate))  # Clamp to [0,1]
        self.efficiency_variance = max(0.0, efficiency_variance)
        self.max_concurrent_jobs = max(1, max_concurrent_jobs)
        
        self.status = WorkerStatus.IDLE
        self.current_jobs: List[Job] = []
        self.completed_jobs = 0
        self.failed_jobs = 0
        self.total_processing_time = 0.0
        self.created_at = datetime.now()
        self.last_job_completed_at: Optional[datetime] = None
        
        self._lock = threading.Lock()
    
    def can_accept_job(self) -> bool:
        """
        Check if worker can accept a new job.
        
        Returns:
            True if worker can accept more jobs, False otherwise
        """
        # Removed threading lock for now
        return (self.status != WorkerStatus.OFFLINE and 
                len(self.current_jobs) < self.max_concurrent_jobs)
    
    def assign_job(self, job: Job) -> bool:
        """
        Assign a job to this worker.
        
        Args:
            job: Job to assign to this worker
            
        Returns:
            True if job was successfully assigned, False otherwise
        """
        # Removed threading lock for now to debug hanging issue
        if not self.can_accept_job():
            return False
        
        job.start_processing()
        self.current_jobs.append(job)
        
        if len(self.current_jobs) == 1:
            self.status = WorkerStatus.BUSY
        
        return True
    
    def process_jobs(self, simulation_time: float) -> List[Job]:
        """
        Process assigned jobs based on simulation time and worker characteristics.
        
        Args:
            simulation_time: Current simulation time
            
        Returns:
            List of completed jobs (successful or failed)
        """
        completed_jobs = []
        
        # Removed threading lock for now
        jobs_to_remove = []
        
        for job in self.current_jobs:
            if self._should_complete_job(job, simulation_time):
                if self._should_fail_job():
                    job.fail_processing(f"Worker {self.worker_id} processing failure")
                    self.failed_jobs += 1
                else:
                    job.complete_processing()
                    self.completed_jobs += 1
                    self.total_processing_time += job.get_processing_time() or 0
                
                completed_jobs.append(job)
                jobs_to_remove.append(job)
                self.last_job_completed_at = datetime.now()
        
        # Remove completed jobs
        for job in jobs_to_remove:
            self.current_jobs.remove(job)
        
        # Update worker status
        if not self.current_jobs:
            self.status = WorkerStatus.IDLE
        
        return completed_jobs
    
    def _should_complete_job(self, job: Job, simulation_time: float) -> bool:
        """
        Determine if a job should be completed based on processing characteristics.
        
        Args:
            job: The job being processed
            simulation_time: Current simulation time
            
        Returns:
            True if job should be completed
        """
        if not job.started_at:
            return False
        
        # Calculate effective processing time with worker characteristics
        base_duration = job.duration
        
        # Apply processing speed
        adjusted_duration = base_duration / self.processing_speed
        
        # Apply efficiency variance (normal distribution around 1.0)
        efficiency_factor = random.normalvariate(1.0, self.efficiency_variance)
        efficiency_factor = max(0.1, efficiency_factor)  # Minimum efficiency
        
        final_duration = adjusted_duration * efficiency_factor
        
        # Store the adjusted duration in job metadata for simulation tracking
        if "adjusted_duration" not in job.metadata:
            job.metadata["adjusted_duration"] = final_duration
            job.metadata["simulation_start_time"] = simulation_time
        
        # Use simulation time instead of real time
        elapsed_simulation_time = simulation_time - job.metadata.get("simulation_start_time", simulation_time)
        return elapsed_simulation_time >= job.metadata["adjusted_duration"]
    
    def _should_fail_job(self) -> bool:
        """
        Determine if a job should fail based on worker failure rate.
        
        Returns:
            True if job should fail
        """
        return random.random() < self.failure_rate
    
    def set_offline(self) -> List[Job]:
        """
        Set worker offline and return any jobs currently being processed.
        
        Returns:
            List of jobs that were interrupted
        """
        # Removed threading lock for now
        self.status = WorkerStatus.OFFLINE
        interrupted_jobs = self.current_jobs.copy()
        self.current_jobs.clear()
        
        # Reset interrupted jobs to pending status
        for job in interrupted_jobs:
            job.status = JobStatus.PENDING
            job.started_at = None
        
        return interrupted_jobs
    
    def set_online(self) -> None:
        """Set worker back online and available for work."""
        # Removed threading lock for now
        if self.status == WorkerStatus.OFFLINE:
            self.status = WorkerStatus.IDLE
    
    def get_utilization(self) -> float:
        """
        Get worker utilization rate (0.0 to 1.0).
        
        Returns:
            Current utilization based on assigned jobs
        """
        # Removed threading lock for now
        if self.max_concurrent_jobs == 0:
            return 0.0
        return len(self.current_jobs) / self.max_concurrent_jobs
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive worker statistics.
        
        Returns:
            Dictionary containing worker performance metrics
        """
        # Removed threading lock for now
        total_jobs = self.completed_jobs + self.failed_jobs
        uptime = (datetime.now() - self.created_at).total_seconds()
        
        return {
            "worker_id": self.worker_id,
            "status": self.status.value,
            "processing_speed": self.processing_speed,
            "failure_rate": self.failure_rate,
            "efficiency_variance": self.efficiency_variance,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "current_jobs": len(self.current_jobs),
            "utilization": self.get_utilization(),
            "completed_jobs": self.completed_jobs,
            "failed_jobs": self.failed_jobs,
            "total_jobs_processed": total_jobs,
            "success_rate": self.completed_jobs / total_jobs if total_jobs > 0 else 0.0,
            "average_processing_time": (
                self.total_processing_time / self.completed_jobs 
                if self.completed_jobs > 0 else 0.0
            ),
            "uptime_seconds": uptime,
            "last_job_completed_at": (
                self.last_job_completed_at.isoformat() 
                if self.last_job_completed_at else None
            )
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert worker to dictionary for serialization."""
        return self.get_stats()


class DistributionType(Enum):
    """Enumeration of supported statistical distributions."""
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    EXPONENTIAL = "exponential"
    UNIFORM = "uniform"
    GAMMA = "gamma"
    POISSON = "poisson"
    NHPP = "nhpp"  # Non-Homogeneous Poisson Process
    CONSTANT = "constant"


class JobGenerator:
    """
    Generates random values based on statistical distributions using scipy.stats.
    
    Provides methods to generate job attributes (size, duration, inter-arrival times)
    using various probability distributions with enhanced statistical capabilities.
    """
    
    @staticmethod
    def _ensure_distribution_type(distribution) -> DistributionType:
        """
        Convert string or DistributionType to DistributionType enum.
        
        Args:
            distribution: String or DistributionType
            
        Returns:
            DistributionType enum
        """
        if isinstance(distribution, str):
            try:
                return DistributionType(distribution.lower())
            except ValueError:
                raise ValueError(f"Unknown distribution string: {distribution}")
        elif isinstance(distribution, DistributionType):
            return distribution
        else:
            raise ValueError(f"Distribution must be string or DistributionType, got: {type(distribution)}")
    
    @staticmethod
    def generate_value(distribution, **params) -> float:
        """
        Generate a random value based on the specified distribution using scipy.stats.
        
        Args:
            distribution: Type of distribution to use (string or DistributionType)
            **params: Distribution-specific parameters
            
        Returns:
            Generated random value
        """
        # Convert to DistributionType if needed
        # Convert to DistributionType if needed
        dist_type = JobGenerator._ensure_distribution_type(distribution)
        
        if dist_type == DistributionType.NORMAL:
            mu = params.get('mean', 1.0)
            sigma = max(0.01, params.get('std', 0.1))  # Ensure positive std
            return max(0.01, stats.norm.rvs(loc=mu, scale=sigma))
        
        elif dist_type == DistributionType.LOGNORMAL:
            # For lognormal: if X ~ LogNormal(μ, σ²), then ln(X) ~ Normal(μ, σ²)
            mu = params.get('mean_log', 0.0)
            sigma = max(0.01, params.get('std_log', 0.5))  # Ensure positive std
            return stats.lognorm.rvs(s=sigma, scale=np.exp(mu))
        
        elif dist_type == DistributionType.EXPONENTIAL:
            # scipy uses scale parameter (1/rate), not rate parameter
            rate = max(0.01, params.get('lambda', 1.0))  # Ensure positive rate
            scale = 1.0 / rate
            return stats.expon.rvs(scale=scale)
        
        elif dist_type == DistributionType.UNIFORM:
            low = params.get('low', 0.0)
            high = params.get('high', 2.0)
            # Ensure high > low
            if high <= low:
                high = low + 1.0
            return stats.uniform.rvs(loc=low, scale=high-low)
        
        elif dist_type == DistributionType.GAMMA:
            # scipy gamma uses shape (a) and scale parameters
            alpha = max(0.01, params.get('alpha', 2.0))  # Ensure positive shape
            beta = max(0.01, params.get('beta', 1.0))    # Ensure positive scale
            return stats.gamma.rvs(a=alpha, scale=beta)
        
        elif dist_type == DistributionType.POISSON:
            mu = max(0.01, params.get('mu', 1.0))  # Ensure positive rate
            # For continuous approximation, use normal distribution
            return max(0.01, stats.norm.rvs(loc=mu, scale=np.sqrt(mu)))
        
        elif dist_type == DistributionType.NHPP:
            # Non-Homogeneous Poisson Process
            # Intensity function: λ(t) = base_rate * (1 + peak_factor * sin(2π * t / period + phase))
            current_time = params.get('current_time', 0.0)
            base_rate = params.get('base_rate', 1.0)
            peak_factor = params.get('peak_factor', 0.5)  # Amplitude of variation (0-1)
            period = params.get('period', 24.0)  # Period in time units (e.g., 24 hours)
            phase = params.get('phase', 0.0)  # Phase shift
            
            # Validate parameters
            base_rate = max(0.01, base_rate)  # Ensure positive base rate
            peak_factor = max(0.0, min(1.0, peak_factor))  # Clamp to [0, 1]
            period = max(0.1, period)  # Ensure positive period
            
            # Calculate time-dependent intensity using the actual simulation time
            # This creates a sinusoidal pattern over the specified period
            time_in_cycle = current_time % period  # Get position within the current cycle
            normalized_time = time_in_cycle / period  # Normalize to [0, 1)
            
            # Apply sinusoidal variation: λ(t) = base_rate * (1 + peak_factor * sin(2π * t + phase))
            sine_component = np.sin(2 * np.pi * normalized_time + phase)
            intensity = base_rate * (1 + peak_factor * sine_component)
            intensity = max(0.01, intensity)  # Ensure positive intensity
            
            # Generate inter-arrival time using exponential with time-dependent rate
            inter_arrival = stats.expon.rvs(scale=1.0/intensity)
            
            # Debug logging for pattern verification (can be removed later)
            if hasattr(params, 'debug') and params.get('debug', False):
                print(f"NHPP Debug: t={current_time:.2f}, period={period}, "
                      f"normalized_t={normalized_time:.3f}, intensity={intensity:.3f}, "
                      f"inter_arrival={inter_arrival:.3f}")
            
            return inter_arrival
        
        elif dist_type == DistributionType.CONSTANT:
            return params.get('value', 1.0)
        
        else:
            raise ValueError(f"Unsupported distribution: {dist_type}")
    
    @staticmethod
    def get_distribution_info(distribution: DistributionType) -> Dict[str, Any]:
        """
        Get information about a distribution including parameter descriptions.
        
        Args:
            distribution: Type of distribution
            
        Returns:
            Dictionary with distribution information
        """
        info = {
            DistributionType.NORMAL: {
                "name": "Normal Distribution",
                "parameters": {
                    "mean": "Mean (μ) - center of distribution",
                    "std": "Standard deviation (σ) - spread of distribution"
                },
                "use_cases": ["Job processing times", "Worker efficiency variations"]
            },
            DistributionType.LOGNORMAL: {
                "name": "Log-Normal Distribution", 
                "parameters": {
                    "mean_log": "Mean of underlying normal distribution",
                    "std_log": "Standard deviation of underlying normal distribution"
                },
                "use_cases": ["Job sizes", "Resource requirements", "File sizes"]
            },
            DistributionType.EXPONENTIAL: {
                "name": "Exponential Distribution",
                "parameters": {
                    "lambda": "Rate parameter (λ) - average rate of events"
                },
                "use_cases": ["Inter-arrival times", "Service times", "Failure times"]
            },
            DistributionType.UNIFORM: {
                "name": "Uniform Distribution",
                "parameters": {
                    "low": "Lower bound",
                    "high": "Upper bound"
                },
                "use_cases": ["Priorities", "Random selections", "Equal probability ranges"]
            },
            DistributionType.GAMMA: {
                "name": "Gamma Distribution",
                "parameters": {
                    "alpha": "Shape parameter (α) - affects curve shape",
                    "beta": "Scale parameter (β) - affects curve scale"
                },
                "use_cases": ["Processing times", "Queue waiting times", "Reliability modeling"]
            },
            DistributionType.POISSON: {
                "name": "Poisson Distribution (Continuous Approximation)",
                "parameters": {
                    "mu": "Average rate (μ) - expected number of events"
                },
                "use_cases": ["Job arrival counts", "Event frequencies", "Count data"]
            },
            DistributionType.NHPP: {
                "name": "Non-Homogeneous Poisson Process",
                "parameters": {
                    "current_time": "Current simulation time",
                    "base_rate": "Base arrival rate (λ₀)",
                    "peak_factor": "Peak variation factor (0-1) - amplitude of rate variation",
                    "period": "Period of rate variation (e.g., 24.0 for daily cycle)",
                    "phase": "Phase shift in radians (0 for peak at t=period/4)"
                },
                "use_cases": ["Time-dependent job arrivals", "Daily/weekly load patterns", "Peak-hour modeling"]
            },
            DistributionType.CONSTANT: {
                "name": "Constant Value",
                "parameters": {
                    "value": "Fixed value to return"
                },
                "use_cases": ["Fixed processing times", "Deterministic scenarios"]
            }
        }
        
        return info.get(distribution, {"name": "Unknown", "parameters": {}, "use_cases": []})
    
    @staticmethod
    def generate_batch_values(distribution, count: int, **params) -> List[float]:
        """
        Generate a batch of random values from the same distribution.
        
        Args:
            distribution: Type of distribution to use (string or DistributionType)
            count: Number of values to generate
            **params: Distribution-specific parameters
            
        Returns:
            List of generated random values
        """
        return [JobGenerator.generate_value(distribution, **params) for _ in range(count)]
    
    @staticmethod
    def get_distribution_stats(values: List[float]) -> Dict[str, float]:
        """
        Calculate statistics for a list of values.
        
        Args:
            values: List of numeric values
            
        Returns:
            Dictionary containing statistical measures
        """
        if not values:
            return {}
        
        np_values = np.array(values)
        return {
            "count": len(values),
            "mean": float(np.mean(np_values)),
            "std": float(np.std(np_values)),
            "min": float(np.min(np_values)),
            "max": float(np.max(np_values)),
            "median": float(np.median(np_values)),
            "q25": float(np.percentile(np_values, 25)),
            "q75": float(np.percentile(np_values, 75)),
            "variance": float(np.var(np_values)),
            "skewness": float(stats.skew(np_values)),
            "kurtosis": float(stats.kurtosis(np_values))
        }
    
    @staticmethod
    def validate_distribution_params(distribution, params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate parameters for a given distribution.
        
        Args:
            distribution: Type of distribution (string or DistributionType)
            params: Parameters to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            dist_type = JobGenerator._ensure_distribution_type(distribution)
            
            if dist_type == DistributionType.NORMAL:
                mean = params.get('mean', 1.0)
                std = params.get('std', 0.1)
                if std <= 0:
                    return False, "Standard deviation must be positive"
                    
            elif dist_type == DistributionType.LOGNORMAL:
                std_log = params.get('std_log', 0.5)
                if std_log <= 0:
                    return False, "Log standard deviation must be positive"
                    
            elif dist_type == DistributionType.EXPONENTIAL:
                lambd = params.get('lambda', 1.0)
                if lambd <= 0:
                    return False, "Lambda (rate) parameter must be positive"
                    
            elif dist_type == DistributionType.UNIFORM:
                low = params.get('low', 0.0)
                high = params.get('high', 2.0)
                if low >= high:
                    return False, "High bound must be greater than low bound"
                    
            elif dist_type == DistributionType.GAMMA:
                alpha = params.get('alpha', 2.0)
                beta = params.get('beta', 1.0)
                if alpha <= 0 or beta <= 0:
                    return False, "Alpha and beta parameters must be positive"
                    
            elif dist_type == DistributionType.POISSON:
                mu = params.get('mu', 1.0)
                if mu <= 0:
                    return False, "Mu parameter must be positive"
            
            elif dist_type == DistributionType.NHPP:
                base_rate = params.get('base_rate', 1.0)
                peak_factor = params.get('peak_factor', 0.5)
                period = params.get('period', 24.0)
                if base_rate <= 0:
                    return False, "Base rate must be positive"
                if not (0 <= peak_factor <= 1):
                    return False, "Peak factor must be between 0 and 1"
                if period <= 0:
                    return False, "Period must be positive"
            
            # Test generation to ensure parameters work
            JobGenerator.generate_value(distribution, **params)
            return True, ""
            
        except Exception as e:
            return False, f"Parameter validation failed: {str(e)}"
    
    @staticmethod
    def create_distribution_config(
        distribution, 
        **params
    ) -> Dict[str, Any]:
        """
        Create a standardized distribution configuration.
        
        Args:
            distribution: Type of distribution (string or DistributionType)
            **params: Distribution parameters
            
        Returns:
            Standardized configuration dictionary
        """
        dist_type = JobGenerator._ensure_distribution_type(distribution)
        is_valid, error_msg = JobGenerator.validate_distribution_params(dist_type, params)
        if not is_valid:
            raise ValueError(f"Invalid distribution parameters: {error_msg}")
        
        return {
            "distribution": dist_type,
            "params": params,
            "info": JobGenerator.get_distribution_info(dist_type)
        }


class WorkFactory:
    """
    Factory for generating jobs based on statistical distributions and scenarios.
    
    Provides configurable job generation with realistic workload patterns,
    including job size, duration, type, and arrival patterns.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the work factory with configuration.
        
        Args:
            config: Configuration dictionary containing distribution parameters
        """
        default_config = self._get_default_config()
        if config:
            # Merge user config with defaults
            merged_config = default_config.copy()
            merged_config.update(config)
            self.config = merged_config
        else:
            self.config = default_config
        
        self.generator = JobGenerator()
        self.jobs_created = 0
        self._lock = threading.Lock()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for job generation."""
        return {
            "job_size": {
                "distribution": "lognormal",
                "params": {"mean_log": 0.0, "std_log": 0.5}
            },
            "job_duration": {
                "distribution": "gamma",
                "params": {"alpha": 2.0, "beta": 1.0}
            },
            "job_priority": {
                "distribution": "uniform",
                "params": {"low": 0, "high": 10}
            },
            "inter_arrival_time": {
                "distribution": "exponential",
                "params": {"lambda": 1.0}
            },
            "job_types": {
                JobType.MILP: 0.3,
                JobType.HEURISTIC: 0.4,
                JobType.ML: 0.2,
                JobType.MIXED: 0.1
            },
            "failure_probability": 0.01
        }
    
    def create_job(self, override_params: Dict[str, Any] = None) -> Job:
        """
        Create a new job with randomly generated attributes.
        
        Args:
            override_params: Optional parameters to override defaults
            
        Returns:
            Newly created job with generated attributes
        """
        with self._lock:
            # Generate job attributes
            size = self._generate_job_size(override_params)
            duration = self._generate_job_duration(override_params)
            priority = self._generate_job_priority(override_params)
            job_type = self._generate_job_type(override_params)
            
            # Create job
            job = Job(
                job_type=job_type,
                size=size,
                duration=duration,
                priority=int(priority),
                metadata={
                    "generated_by": "WorkFactory",
                    "generation_id": self.jobs_created
                }
            )
            
            # Apply any additional overrides
            if override_params:
                for key, value in override_params.items():
                    if hasattr(job, key):
                        setattr(job, key, value)
            
            self.jobs_created += 1
            return job
    
    def create_batch(self, count: int, override_params: Dict[str, Any] = None) -> List[Job]:
        """
        Create a batch of jobs.
        
        Args:
            count: Number of jobs to create
            override_params: Optional parameters to override defaults
            
        Returns:
            List of created jobs
        """
        return [self.create_job(override_params) for _ in range(count)]
    
    def generate_inter_arrival_time(self, current_time: float = 0.0) -> float:
        """
        Generate the time until the next job arrives.
        
        Args:
            current_time: Current simulation time (needed for NHPP and time-dependent distributions)
        
        Returns:
            Time in seconds until next job arrival
        """
        config = self.config["inter_arrival_time"]
        
        # Prepare parameters for distribution generation
        params = config["params"].copy()
        
        # Add current_time for time-dependent distributions (like NHPP)
        if config["distribution"] in ["nhpp", "NHPP", DistributionType.NHPP]:
            params["current_time"] = current_time
        
        inter_arrival_time = self.generator.generate_value(
            config["distribution"], 
            **params
        )
        
        # Ensure reasonable bounds for inter-arrival time
        # Minimum 0.01 seconds, maximum 60 seconds
        inter_arrival_time = max(0.01, min(inter_arrival_time, 60.0))
        
        return inter_arrival_time
    
    def generate_scenario_arrivals(self, scenario_duration_seconds: float) -> List[float]:
        """
        Generate all job arrival times for the entire scenario duration.
        
        Args:
            scenario_duration_seconds: Total scenario duration in seconds
            
        Returns:
            Sorted list of arrival times (in seconds from scenario start)
        """
        config = self.config["inter_arrival_time"]
        distribution = config["distribution"]
        params = config["params"]
        
        arrival_times = []
        current_time = 0.0
        
        if distribution in ["nhpp", "NHPP", DistributionType.NHPP]:
            # Use Non-Homogeneous Poisson Process with time-varying rates
            arrival_times = self._generate_nhpp_arrivals(scenario_duration_seconds)
        else:
            # Use standard inter-arrival time generation
            while current_time < scenario_duration_seconds:
                inter_arrival = self.generate_inter_arrival_time(current_time)
                current_time += inter_arrival
                if current_time < scenario_duration_seconds:
                    arrival_times.append(current_time)
        
        return sorted(arrival_times)
    
    def _generate_nhpp_arrivals(self, scenario_duration_seconds: float) -> List[float]:
        """
        Generate arrivals using Non-Homogeneous Poisson Process with 24-hour patterns.
        
        Args:
            scenario_duration_seconds: Total scenario duration in seconds
            
        Returns:
            List of arrival times
        """
        config = self.config["inter_arrival_time"]["params"]
        base_rate = config.get("base_rate", 1.0)  # jobs per second base rate
        peak_rate = config.get("peak_rate", 3.0)   # jobs per second peak rate
        peak_hours = config.get("peak_hours", [(9, 17)])  # Default: 9 AM to 5 PM
        
        arrivals = []
        current_time = 0.0
        
        while current_time < scenario_duration_seconds:
            # Calculate current hour of day (0-23)
            hour_of_day = (current_time / 3600) % 24
            
            # Determine current intensity based on time of day
            intensity = self._get_hourly_intensity(hour_of_day, base_rate, peak_rate, peak_hours)
            
            # Generate next inter-arrival time using current intensity
            if intensity > 0:
                inter_arrival = np.random.exponential(1.0 / intensity)
            else:
                inter_arrival = 60.0  # 1 minute if no intensity
            
            current_time += inter_arrival
            
            if current_time < scenario_duration_seconds:
                arrivals.append(current_time)
        
        return arrivals
    
    def _get_hourly_intensity(self, hour_of_day: float, base_rate: float, 
                             peak_rate: float, peak_hours: List[Tuple[int, int]]) -> float:
        """
        Calculate job arrival intensity based on hour of day.
        
        Args:
            hour_of_day: Current hour (0-23)
            base_rate: Base arrival rate (jobs per second)
            peak_rate: Peak arrival rate (jobs per second) 
            peak_hours: List of (start_hour, end_hour) tuples for peak periods
            
        Returns:
            Current arrival intensity (jobs per second)
        """
        # Check if we're in a peak period
        for start_hour, end_hour in peak_hours:
            if start_hour <= hour_of_day < end_hour:
                # Linear interpolation for smooth transitions
                if hour_of_day < start_hour + 1:  # First hour of peak
                    transition = (hour_of_day - start_hour)
                    return base_rate + (peak_rate - base_rate) * transition
                elif hour_of_day >= end_hour - 1:  # Last hour of peak
                    transition = (end_hour - hour_of_day)
                    return base_rate + (peak_rate - base_rate) * transition
                else:  # Full peak
                    return peak_rate
        
        # Off-peak hours - reduce to fraction of base rate
        if 22 <= hour_of_day or hour_of_day < 6:  # Night hours (10 PM - 6 AM)
            return base_rate * 0.1
        elif 6 <= hour_of_day < 9 or 17 <= hour_of_day < 22:  # Transition hours
            return base_rate * 0.5
        else:
            return base_rate
    
    def _generate_job_size(self, override_params: Dict[str, Any] = None) -> float:
        """Generate job size based on configuration."""
        if override_params and "size" in override_params:
            return override_params["size"]
        
        config = self.config["job_size"]
        return self.generator.generate_value(
            config["distribution"], 
            **config["params"]
        )
    
    def _generate_job_duration(self, override_params: Dict[str, Any] = None) -> float:
        """Generate job duration based on configuration."""
        if override_params and "duration" in override_params:
            return override_params["duration"]
        
        config = self.config["job_duration"]
        return self.generator.generate_value(
            config["distribution"], 
            **config["params"]
        )
    
    def _generate_job_priority(self, override_params: Dict[str, Any] = None) -> float:
        """Generate job priority based on configuration."""
        if override_params and "priority" in override_params:
            return override_params["priority"]
        
        config = self.config["job_priority"]
        return self.generator.generate_value(
            config["distribution"], 
            **config["params"]
        )
    
    def _generate_job_type(self, override_params: Dict[str, Any] = None) -> JobType:
        """Generate job type based on probability distribution."""
        if override_params and "job_type" in override_params:
            job_type = override_params["job_type"]
            if isinstance(job_type, str):
                try:
                    return JobType(job_type)
                except ValueError:
                    return JobType.MIXED  # fallback
            return job_type
        
        # Weighted random selection
        job_types_config = self.config["job_types"]
        
        # Convert string keys to JobType enums if necessary
        job_types = []
        weights = []
        
        for key, weight in job_types_config.items():
            if isinstance(key, str):
                try:
                    job_type = JobType(key)
                except ValueError:
                    job_type = JobType.MIXED  # fallback for unknown types
            else:
                job_type = key  # already a JobType enum
            
            job_types.append(job_type)
            weights.append(weight)
        
        rand_val = random.random()
        cumulative = 0.0
        
        for job_type, weight in zip(job_types, weights):
            cumulative += weight
            if rand_val <= cumulative:
                return job_type
        
        # Fallback to first type
        return job_types[0] if job_types else JobType.MIXED
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update the factory configuration.
        
        Args:
            new_config: New configuration to merge with existing config
        """
        with self._lock:
            self.config.update(new_config)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get factory statistics.
        
        Returns:
            Dictionary containing generation statistics
        """
        with self._lock:
            return {
                "jobs_created": self.jobs_created,
                "config": self.config
            }


class SimulationLogger:
    """
    Handles logging of simulation data to JSON and CSV formats.
    
    Provides structured logging for simulation state, job processing,
    worker utilization, and queue statistics over time with enhanced
    persistence and data integrity features.
    """
    
    def __init__(self, log_directory: str = "logs"):
        """
        Initialize the simulation logger.
        
        Args:
            log_directory: Directory to store log files
        """
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(exist_ok=True)
        
        # Initialize log file paths
        self.json_log_file = self.log_directory / "simulation_log.json"
        self.csv_log_file = self.log_directory / "simulation_log.csv"
        self.jobs_csv_file = self.log_directory / "jobs_log.csv"
        self.workers_csv_file = self.log_directory / "workers_log.csv"
        self.detailed_log_file = self.log_directory / "detailed_simulation_log.json"
        self.metrics_log_file = self.log_directory / "metrics_log.csv"
        
        # Initialize log structures
        self.json_logs: List[Dict[str, Any]] = []
        self.detailed_logs: List[Dict[str, Any]] = []
        self.csv_headers_written = False
        self.metrics_headers_written = False
        
        # Data integrity and persistence features
        self.log_buffer_size = 100  # Buffer size before flushing to disk
        self.current_buffer_size = 0
        self.last_flush_time = time.time()
        self.flush_interval = 5.0  # Flush every 5 seconds
        
        # Clear existing logs
        self._initialize_log_files()
    
    def _initialize_log_files(self) -> None:
        """Initialize log files with headers."""
        # Initialize JSON log
        self.json_logs = []
        self.detailed_logs = []
        
        # Initialize CSV headers
        csv_headers = [
            "timestamp", "simulation_time", "total_jobs_created", "jobs_in_queue",
            "jobs_completed", "jobs_failed", "total_workers", "active_workers",
            "average_queue_length", "average_worker_utilization", "throughput"
        ]
        
        jobs_headers = [
            "timestamp", "job_id", "job_type", "size", "duration", "priority",
            "status", "created_at", "started_at", "completed_at", "processing_time",
            "worker_id"
        ]
        
        workers_headers = [
            "timestamp", "worker_id", "status", "processing_speed", "failure_rate",
            "current_jobs", "completed_jobs", "failed_jobs", "utilization",
            "success_rate", "average_processing_time"
        ]
        
        metrics_headers = [
            "timestamp", "simulation_time", "metric_name", "metric_value", 
            "metric_type", "metadata"
        ]
        
        # Write CSV headers
        with open(self.csv_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
        
        with open(self.jobs_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(jobs_headers)
        
        with open(self.workers_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(workers_headers)
            
        with open(self.metrics_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(metrics_headers)
        
        # Initialize detailed JSON log file
        with open(self.detailed_log_file, 'w') as f:
            json.dump([], f)
        
        # Initialize main JSON log file
        with open(self.json_log_file, 'w') as f:
            json.dump([], f)
    
    def log_simulation_state(
        self, 
        simulation_time: float,
        queue_manager: QueueManager,
        workers: List[Worker],
        total_jobs_created: int = 0,
        force_flush: bool = False
    ) -> None:
        """
        Log the current simulation state with enhanced persistence features.
        
        Args:
            simulation_time: Current simulation time
            queue_manager: Queue manager instance
            workers: List of worker instances
            total_jobs_created: Total number of jobs created so far
            force_flush: Force immediate flush to disk
        """
        timestamp = datetime.now().isoformat()
        
        # Get queue statistics
        queue_stats = queue_manager.get_queue_stats()
        
        # Calculate worker statistics
        active_workers = len([w for w in workers if w.status == WorkerStatus.BUSY])
        worker_utilizations = [w.get_utilization() for w in workers]
        avg_utilization = sum(worker_utilizations) / len(worker_utilizations) if worker_utilizations else 0.0
        
        # Calculate throughput (jobs completed in recent time window)
        total_completed = sum(w.completed_jobs for w in workers)
        total_failed = sum(w.failed_jobs for w in workers)
        throughput = (total_completed + total_failed) / (simulation_time + 0.001)  # Avoid division by zero
        
        # Create comprehensive log entry
        log_entry = {
            "timestamp": timestamp,
            "simulation_time": simulation_time,
            "total_jobs_created": total_jobs_created,
            "queue_stats": queue_stats,
            "worker_stats": {
                "total_workers": len(workers),
                "active_workers": active_workers,
                "average_utilization": avg_utilization,
                "individual_utilizations": worker_utilizations,
                "individual_stats": [w.get_stats() for w in workers]
            },
            "performance_metrics": {
                "throughput": throughput,
                "total_completed": total_completed,
                "total_failed": total_failed,
                "success_rate": total_completed / (total_completed + total_failed) if (total_completed + total_failed) > 0 else 0.0
            }
        }
        
        # Create detailed log entry with individual job and worker states
        detailed_entry = {
            **log_entry,
            "detailed_worker_states": [w.to_dict() for w in workers],
            "queue_contents": self._get_queue_contents_for_logging(queue_manager)
        }
        
        # Add to logs
        self.json_logs.append(log_entry)
        self.detailed_logs.append(detailed_entry)
        self.current_buffer_size += 1
        
        # Log additional metrics
        self._log_metrics(simulation_time, {
            "queue_length": queue_stats["queue_length"],
            "active_workers": active_workers,
            "throughput": throughput,
            "success_rate": log_entry["performance_metrics"]["success_rate"],
            "average_utilization": avg_utilization
        })
        
        # Write to CSV immediately for real-time monitoring
        csv_row = [
            timestamp, simulation_time, total_jobs_created, queue_stats["queue_length"],
            queue_stats["completed_jobs"], queue_stats["failed_jobs"], len(workers),
            active_workers, queue_stats["queue_length"], avg_utilization, throughput
        ]
        
        with open(self.csv_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_row)
        
        # Check if we need to flush to disk
        current_time = time.time()
        if (force_flush or 
            self.current_buffer_size >= self.log_buffer_size or 
            current_time - self.last_flush_time >= self.flush_interval):
            self._flush_logs()
    
    def _get_queue_contents_for_logging(self, queue_manager: QueueManager) -> List[Dict[str, Any]]:
        """
        Safely extract queue contents for logging, handling different queue types.
        
        Args:
            queue_manager: Queue manager instance
            
        Returns:
            List of job information dictionaries for logging
        """
        queue_contents = []
        
        try:
            if hasattr(queue_manager, '_queue'):
                queue = queue_manager._queue
                
                if queue_manager.use_priority_queue:
                    # For priority queue (heap), we need to be careful about accessing items
                    for i, item in enumerate(list(queue)[:10]):  # First 10 jobs
                        if len(item) >= 3:  # (priority, timestamp, job)
                            job = item[2]
                            queue_contents.append({
                                "position": i,
                                "job_id": getattr(job, 'job_id', 'unknown'),
                                "priority": -item[0],  # Convert back from negative
                                "job_type": getattr(job, 'job_type', 'unknown')
                            })
                else:
                    # For FIFO queue (deque)
                    for i, job in enumerate(list(queue)[:10]):  # First 10 jobs
                        queue_contents.append({
                            "position": i,
                            "job_id": getattr(job, 'job_id', 'unknown'),
                            "job_type": getattr(job, 'job_type', 'unknown')
                        })
        except Exception as e:
            # If anything goes wrong, just return empty list
            queue_contents = [{"error": f"Could not extract queue contents: {e}"}]
        
        return queue_contents
    
    def _log_metrics(self, simulation_time: float, metrics: Dict[str, Any]) -> None:
        """
        Log individual metrics for detailed analysis.
        
        Args:
            simulation_time: Current simulation time
            metrics: Dictionary of metric name-value pairs
        """
        timestamp = datetime.now().isoformat()
        
        with open(self.metrics_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for metric_name, metric_value in metrics.items():
                metric_type = type(metric_value).__name__
                metadata = ""
                if isinstance(metric_value, (int, float)):
                    metadata = f"numeric_{metric_type}"
                
                writer.writerow([
                    timestamp, simulation_time, metric_name, 
                    metric_value, metric_type, metadata
                ])
    
    def _flush_logs(self) -> None:
        """Flush buffered logs to disk for persistence."""
        try:
            # Save JSON logs
            with open(self.json_log_file, 'w') as f:
                json.dump(self.json_logs, f, indent=2, cls=SimulationJSONEncoder)
            
            # Save detailed logs
            with open(self.detailed_log_file, 'w') as f:
                json.dump(self.detailed_logs, f, indent=2, cls=SimulationJSONEncoder)
            
            # Reset buffer
            self.current_buffer_size = 0
            self.last_flush_time = time.time()
            
        except Exception as e:
            print(f"Warning: Failed to flush logs to disk: {e}")
    
    def append_to_json_log_file(self, log_entry: Dict[str, Any]) -> None:
        """
        Append a single log entry to JSON file for real-time persistence.
        
        Args:
            log_entry: Log entry to append
        """
        try:
            # For real-time logging, append to a separate file
            append_log_file = self.log_directory / "realtime_simulation_log.jsonl"
            with open(append_log_file, 'a') as f:
                f.write(json.dumps(log_entry, cls=SimulationJSONEncoder) + '\n')
        except Exception as e:
            print(f"Warning: Failed to append to JSON log file: {e}")
    
    def log_job_event(self, job: Job, event_type: str, worker_id: str = None) -> None:
        """
        Log a job-related event.
        
        Args:
            job: Job instance
            event_type: Type of event (created, started, completed, failed)
            worker_id: ID of worker processing the job (if applicable)
        """
        timestamp = datetime.now().isoformat()
        
        job_data = [
            timestamp, job.job_id, 
            job.job_type.value if hasattr(job.job_type, 'value') else str(job.job_type), 
            job.size, job.duration,
            job.priority, job.status.value,
            job.created_at.isoformat() if job.created_at else None,
            job.started_at.isoformat() if job.started_at else None,
            job.completed_at.isoformat() if job.completed_at else None,
            job.get_processing_time(), worker_id or ""
        ]
        
        with open(self.jobs_csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(job_data)
    
    def log_worker_state(self, worker: Worker) -> None:
        """
        Log current worker state.
        
        Args:
            worker: Worker instance
        """
        timestamp = datetime.now().isoformat()
        stats = worker.get_stats()
        
        worker_data = [
            timestamp, stats["worker_id"], stats["status"], stats["processing_speed"],
            stats["failure_rate"], stats["current_jobs"], stats["completed_jobs"],
            stats["failed_jobs"], stats["utilization"], stats["success_rate"],
            stats["average_processing_time"]
        ]
        
        with open(self.workers_csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(worker_data)
    
    def save_json_logs(self, force_flush: bool = True) -> None:
        """
        Save all JSON logs to file with enhanced error handling.
        
        Args:
            force_flush: Force immediate flush even if buffer isn't full
        """
        if force_flush or self.current_buffer_size > 0:
            self._flush_logs()
    
    def backup_logs(self, backup_suffix: str = None) -> Dict[str, str]:
        """
        Create backup copies of all log files.
        
        Args:
            backup_suffix: Optional suffix for backup files (timestamp if None)
            
        Returns:
            Dictionary mapping original file paths to backup file paths
        """
        if backup_suffix is None:
            backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        backup_paths = {}
        log_files = [
            self.json_log_file,
            self.detailed_log_file,
            self.csv_log_file,
            self.jobs_csv_file,
            self.workers_csv_file,
            self.metrics_log_file
        ]
        
        try:
            for log_file in log_files:
                if log_file.exists():
                    backup_path = log_file.with_suffix(f'.{backup_suffix}{log_file.suffix}')
                    import shutil
                    shutil.copy2(log_file, backup_path)
                    backup_paths[str(log_file)] = str(backup_path)
            
            return backup_paths
        except Exception as e:
            print(f"Warning: Failed to create log backups: {e}")
            return {}
    
    def restore_from_backup(self, backup_suffix: str) -> bool:
        """
        Restore logs from backup files.
        
        Args:
            backup_suffix: Suffix of backup files to restore from
            
        Returns:
            True if restoration was successful, False otherwise
        """
        try:
            log_files = [
                self.json_log_file,
                self.detailed_log_file,
                self.csv_log_file,
                self.jobs_csv_file,
                self.workers_csv_file,
                self.metrics_log_file
            ]
            
            for log_file in log_files:
                backup_path = log_file.with_suffix(f'.{backup_suffix}{log_file.suffix}')
                if backup_path.exists():
                    import shutil
                    shutil.copy2(backup_path, log_file)
            
            # Reload JSON logs into memory
            if self.json_log_file.exists():
                with open(self.json_log_file, 'r') as f:
                    self.json_logs = json.load(f)
            
            if self.detailed_log_file.exists():
                with open(self.detailed_log_file, 'r') as f:
                    self.detailed_logs = json.load(f)
            
            return True
        except Exception as e:
            print(f"Error: Failed to restore from backup: {e}")
            return False
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about logged data.
        
        Returns:
            Dictionary containing log statistics and data quality metrics
        """
        stats = {
            "log_files": {},
            "data_integrity": {},
            "performance_metrics": {},
            "time_range": {}
        }
        
        # File statistics
        log_files = {
            "json_log": self.json_log_file,
            "detailed_log": self.detailed_log_file,
            "csv_log": self.csv_log_file,
            "jobs_log": self.jobs_csv_file,
            "workers_log": self.workers_csv_file,
            "metrics_log": self.metrics_log_file
        }
        
        for name, file_path in log_files.items():
            if file_path.exists():
                file_stat = file_path.stat()
                stats["log_files"][name] = {
                    "path": str(file_path),
                    "size_bytes": file_stat.st_size,
                    "modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                }
        
        # Data integrity checks
        if self.json_logs:
            stats["data_integrity"]["json_entries"] = len(self.json_logs)
            stats["data_integrity"]["detailed_entries"] = len(self.detailed_logs)
            
            # Time range
            first_entry = self.json_logs[0] if self.json_logs else None
            last_entry = self.json_logs[-1] if self.json_logs else None
            
            if first_entry and last_entry:
                stats["time_range"] = {
                    "first_timestamp": first_entry.get("timestamp"),
                    "last_timestamp": last_entry.get("timestamp"),
                    "first_simulation_time": first_entry.get("simulation_time", 0),
                    "last_simulation_time": last_entry.get("simulation_time", 0),
                    "total_simulation_time": last_entry.get("simulation_time", 0) - first_entry.get("simulation_time", 0)
                }
            
            # Performance metrics from last entry
            if last_entry:
                stats["performance_metrics"] = last_entry.get("performance_metrics", {})
        
        return stats
    
    def get_log_summary(self) -> Dict[str, Any]:
        """
        Get a summary of logged data with enhanced metrics.
        
        Returns:
            Dictionary containing log summary statistics
        """
        if not self.json_logs:
            return {
                "message": "No logs available",
                "log_statistics": self.get_log_statistics()
            }
        
        total_entries = len(self.json_logs)
        latest_entry = self.json_logs[-1]
        
        summary = {
            "basic_metrics": {
                "total_log_entries": total_entries,
                "simulation_duration": latest_entry["simulation_time"],
                "final_queue_length": latest_entry["queue_stats"]["queue_length"],
                "total_jobs_processed": latest_entry["queue_stats"]["total_jobs_processed"],
                "final_success_rate": latest_entry["performance_metrics"]["success_rate"],
                "average_throughput": latest_entry["performance_metrics"]["throughput"]
            },
            "log_files": {
                "json_log": str(self.json_log_file),
                "detailed_log": str(self.detailed_log_file),
                "csv_log": str(self.csv_log_file),
                "jobs_log": str(self.jobs_csv_file),
                "workers_log": str(self.workers_csv_file),
                "metrics_log": str(self.metrics_log_file)
            },
            "data_quality": {
                "buffer_status": f"{self.current_buffer_size}/{self.log_buffer_size}",
                "last_flush": datetime.fromtimestamp(self.last_flush_time).isoformat(),
                "detailed_entries": len(self.detailed_logs)
            }
        }
        
        # Add comprehensive statistics
        log_statistics = self.get_log_statistics()
        summary["log_statistics"] = log_statistics
        
        return summary
    
    def clear_logs(self, create_backup: bool = True) -> Dict[str, str]:
        """
        Clear all log data and reinitialize files.
        
        Args:
            create_backup: Whether to create backup copies before clearing
            
        Returns:
            Dictionary of backup file paths if backup was created
        """
        backup_paths = {}
        
        if create_backup:
            backup_paths = self.backup_logs("pre_clear")
        
        # Clear in-memory logs
        self.json_logs = []
        self.detailed_logs = []
        self.current_buffer_size = 0
        self.last_flush_time = time.time()
        
        # Reinitialize files
        self._initialize_log_files()
        
        print(f"Logs cleared and reinitialized in {self.log_directory}")
        if backup_paths:
            print(f"Backup files created: {list(backup_paths.values())}")
        
        return backup_paths


@dataclass
class SimulationSnapshot:
    """
    Represents a complete snapshot of simulation state for persistence and resumption.
    """
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    simulation_time: float = 0.0
    total_jobs_created: int = 0
    next_job_arrival_time: float = 0.0
    last_log_time: float = 0.0
    
    # Arrival schedule for scenario-aware generation
    arrival_schedule: List[float] = field(default_factory=list)
    next_arrival_index: int = 0
    scenario_duration_hours: float = 24.0
    
    # Component states
    queue_state: Dict[str, Any] = field(default_factory=dict)
    workers_state: List[Dict[str, Any]] = field(default_factory=list)
    factory_state: Dict[str, Any] = field(default_factory=dict)
    logger_state: Dict[str, Any] = field(default_factory=dict)
    
    # Configuration snapshots
    simulation_config: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary for serialization."""
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp.isoformat(),
            "simulation_time": self.simulation_time,
            "total_jobs_created": self.total_jobs_created,
            "next_job_arrival_time": self.next_job_arrival_time,
            "last_log_time": self.last_log_time,
            "arrival_schedule": self.arrival_schedule,
            "next_arrival_index": self.next_arrival_index,
            "scenario_duration_hours": self.scenario_duration_hours,
            "queue_state": self.queue_state,
            "workers_state": self.workers_state,
            "factory_state": self.factory_state,
            "logger_state": self.logger_state,
            "simulation_config": self.simulation_config,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationSnapshot':
        """Create snapshot from dictionary."""
        snapshot = cls()
        snapshot.snapshot_id = data.get("snapshot_id", str(uuid.uuid4()))
        snapshot.timestamp = datetime.fromisoformat(data["timestamp"])
        snapshot.simulation_time = data["simulation_time"]
        snapshot.total_jobs_created = data["total_jobs_created"]
        snapshot.next_job_arrival_time = data["next_job_arrival_time"]
        snapshot.last_log_time = data["last_log_time"]
        snapshot.arrival_schedule = data.get("arrival_schedule", [])
        snapshot.next_arrival_index = data.get("next_arrival_index", 0)
        snapshot.scenario_duration_hours = data.get("scenario_duration_hours", 24.0)
        snapshot.queue_state = data["queue_state"]
        snapshot.workers_state = data["workers_state"]
        snapshot.factory_state = data["factory_state"]
        snapshot.logger_state = data["logger_state"]
        snapshot.simulation_config = data["simulation_config"]
        snapshot.metadata = data["metadata"]
        return snapshot
        snapshot.workers_state = data["workers_state"]
        snapshot.factory_state = data["factory_state"]
        snapshot.logger_state = data["logger_state"]
        snapshot.simulation_config = data["simulation_config"]
        snapshot.metadata = data["metadata"]
        return snapshot


class SnapshotManager:
    """
    Manages saving and loading simulation snapshots for pause/resume functionality.
    """
    
    def __init__(self, snapshots_directory: str = "snapshots"):
        """
        Initialize the snapshot manager.
        
        Args:
            snapshots_directory: Directory to store snapshot files
        """
        self.snapshots_directory = Path(snapshots_directory)
        self.snapshots_directory.mkdir(exist_ok=True)
    
    def save_snapshot(self, snapshot: SimulationSnapshot, filename: str = None) -> str:
        """
        Save a simulation snapshot to file.
        
        Args:
            snapshot: Snapshot to save
            filename: Optional filename (auto-generated if None)
            
        Returns:
            Path to saved snapshot file
        """
        if filename is None:
            timestamp_str = snapshot.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp_str}_{snapshot.snapshot_id[:8]}.json"
        
        snapshot_path = self.snapshots_directory / filename
        
        try:
            with open(snapshot_path, 'w') as f:
                json.dump(snapshot.to_dict(), f, indent=2)
            
            return str(snapshot_path)
        except Exception as e:
            raise RuntimeError(f"Failed to save snapshot: {e}")
    
    def load_snapshot(self, filename: str) -> SimulationSnapshot:
        """
        Load a simulation snapshot from file.
        
        Args:
            filename: Snapshot filename or full path
            
        Returns:
            Loaded simulation snapshot
        """
        if not Path(filename).is_absolute():
            snapshot_path = self.snapshots_directory / filename
        else:
            snapshot_path = Path(filename)
        
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot file not found: {snapshot_path}")
        
        try:
            with open(snapshot_path, 'r') as f:
                data = json.load(f)
            
            return SimulationSnapshot.from_dict(data)
        except Exception as e:
            raise RuntimeError(f"Failed to load snapshot: {e}")
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """
        List available snapshots with metadata.
        
        Returns:
            List of snapshot information dictionaries
        """
        snapshots = []
        
        for snapshot_file in self.snapshots_directory.glob("*.json"):
            try:
                file_stat = snapshot_file.stat()
                
                # Try to read basic info from file
                with open(snapshot_file, 'r') as f:
                    data = json.load(f)
                
                snapshots.append({
                    "filename": snapshot_file.name,
                    "snapshot_id": data.get("snapshot_id", "unknown"),
                    "timestamp": data.get("timestamp", "unknown"),
                    "simulation_time": data.get("simulation_time", 0),
                    "total_jobs_created": data.get("total_jobs_created", 0),
                    "file_size": file_stat.st_size,
                    "file_modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                })
            except Exception:
                # Skip invalid files
                continue
        
        # Sort by timestamp
        snapshots.sort(key=lambda x: x["timestamp"], reverse=True)
        return snapshots
    
    def delete_snapshot(self, filename: str) -> bool:
        """
        Delete a snapshot file.
        
        Args:
            filename: Snapshot filename
            
        Returns:
            True if deletion was successful, False otherwise
        """
        snapshot_path = self.snapshots_directory / filename
        
        try:
            if snapshot_path.exists():
                snapshot_path.unlink()
                return True
            return False
        except Exception as e:
            print(f"Warning: Failed to delete snapshot {filename}: {e}")
            return False
    
    def cleanup_old_snapshots(self, keep_count: int = 10) -> List[str]:
        """
        Clean up old snapshots, keeping only the most recent ones.
        
        Args:
            keep_count: Number of recent snapshots to keep
            
        Returns:
            List of deleted snapshot filenames
        """
        snapshots = self.list_snapshots()
        
        if len(snapshots) <= keep_count:
            return []
        
        deleted_files = []
        snapshots_to_delete = snapshots[keep_count:]
        
        for snapshot_info in snapshots_to_delete:
            if self.delete_snapshot(snapshot_info["filename"]):
                deleted_files.append(snapshot_info["filename"])
        
        return deleted_files


class SimulationEngine:
    """
    Main simulation engine that orchestrates job generation, queue management,
    worker processing, and logging.
    
    Provides high-level interface for running simulations with configurable
    scenarios and comprehensive data collection.
    """
    
    def __init__(
        self,
        num_workers: int = 4,
        queue_config: Dict[str, Any] = None,
        worker_config: Dict[str, Any] = None,
        factory_config: Dict[str, Any] = None,
        log_directory: str = "logs",
        termination_config: Dict[str, Any] = None,
        scenario_duration_hours: float = 24.0
    ):
        """
        Initialize the simulation engine.
        
        Args:
            num_workers: Number of worker instances to create
            queue_config: Configuration for queue manager
            worker_config: Configuration for worker creation
            factory_config: Configuration for job factory
            log_directory: Directory for log files
            termination_config: Configuration for simulation termination criteria
            scenario_duration_hours: Duration of the scenario in hours (default: 24 hours)
        """
        # Initialize components
        self.queue_manager = QueueManager(**(queue_config or {}))
        self.workers = [
            Worker(**(worker_config or {})) for _ in range(num_workers)
        ]
        self.job_factory = WorkFactory(factory_config)
        self.logger = SimulationLogger(log_directory)
        self.snapshot_manager = SnapshotManager("snapshots")
        
        # Scenario configuration
        self.scenario_duration_hours = scenario_duration_hours
        self.scenario_duration_seconds = scenario_duration_hours * 3600
        
        # Pre-generate arrival schedule for entire scenario
        print(f"Generating arrival schedule for {scenario_duration_hours}-hour scenario...")
        self.arrival_schedule = self.job_factory.generate_scenario_arrivals(
            self.scenario_duration_seconds
        )
        self.next_arrival_index = 0
        print(f"Generated {len(self.arrival_schedule)} job arrivals for scenario")
        
        # Simulation state
        self.simulation_time = 0.0
        self.running = False
        self.paused = False
        self.total_jobs_created = 0
        
        # Legacy support for next_job_arrival_time (now derived from schedule)
        self.next_job_arrival_time = (
            self.arrival_schedule[0] if self.arrival_schedule else float('inf')
        )
        
        # Termination configuration - needs to be set first
        self.termination_config = termination_config or {}
        
        # Configuration
        self.time_step = 0.1  # Simulation time step in seconds
        self.log_interval = self.termination_config.get('log_interval', 60.0)  # How often to log state (default: every minute)
        self.last_log_time = 0.0
        
        # Termination criteria
        self.max_data_points = self.termination_config.get('max_data_points', 1440)  # Default: 24 hours of minute-by-minute data
        self.max_jobs = self.termination_config.get('max_jobs', None)
        self.max_completed_jobs = self.termination_config.get('max_completed_jobs', None)
        self.termination_mode = self.termination_config.get('mode', 'data_points')  # Default to data_points mode
        self.data_points_logged = 0
    
    def create_job(self, override_params: Dict[str, Any] = None) -> Job:
        """
        Create a new job and add it to the queue.
        
        Args:
            override_params: Optional parameters to override defaults
            
        Returns:
            Created job instance
        """
        job = self.job_factory.create_job(override_params)
        
        if self.queue_manager.add_job(job):
            self.total_jobs_created += 1
            self.logger.log_job_event(job, "created")
            return job
        else:
            job.fail_processing("Queue is full")
            self.logger.log_job_event(job, "rejected")
            return job
    
    def step_simulation(self) -> Dict[str, Any]:
        """
        Advance simulation by one time step.
        
        Returns:
            Dictionary containing step results
        """
        # Generate new jobs based on pre-calculated arrival schedule
        jobs_generated_this_step = 0
        while (self.next_arrival_index < len(self.arrival_schedule) and 
               self.arrival_schedule[self.next_arrival_index] <= self.simulation_time):
            
            self.create_job()
            self.next_arrival_index += 1
            jobs_generated_this_step += 1
            
            # Safety check to prevent infinite loops
            if jobs_generated_this_step > 1000:
                print(f"Warning: Generated {jobs_generated_this_step} jobs in one step, breaking")
                break
        
        # Update next_job_arrival_time for compatibility/debugging
        if self.next_arrival_index < len(self.arrival_schedule):
            self.next_job_arrival_time = self.arrival_schedule[self.next_arrival_index]
        else:
            self.next_job_arrival_time = float('inf')  # No more arrivals
        
        # Assign jobs to available workers
        assigned_jobs = []
        for worker in self.workers:
            if worker.can_accept_job():
                job = self.queue_manager.get_next_job()
                if job:
                    if worker.assign_job(job):
                        assigned_jobs.append(job)
                        self.logger.log_job_event(job, "started", worker.worker_id)
        
        # Process jobs on all workers
        completed_jobs = []
        for worker in self.workers:
            worker_completed = worker.process_jobs(self.simulation_time)
            for job in worker_completed:
                if job.status == JobStatus.COMPLETED:
                    self.queue_manager.complete_job(job)
                    self.logger.log_job_event(job, "completed", worker.worker_id)
                else:
                    self.queue_manager.fail_job(job, job.metadata.get("error", ""))
                    self.logger.log_job_event(job, "failed", worker.worker_id)
            completed_jobs.extend(worker_completed)
        
        # Log simulation state periodically
        if self.simulation_time - self.last_log_time >= self.log_interval:
            self.logger.log_simulation_state(
                self.simulation_time, self.queue_manager, 
                self.workers, self.total_jobs_created
            )
            # Also log individual worker states
            for worker in self.workers:
                self.logger.log_worker_state(worker)
            self.last_log_time = self.simulation_time
            self.data_points_logged += 1  # Track data points for termination criteria
        
        # Advance time
        self.simulation_time += self.time_step
        
        return {
            "simulation_time": self.simulation_time,
            "jobs_generated_this_step": jobs_generated_this_step,
            "jobs_assigned": len(assigned_jobs),
            "jobs_completed": len(completed_jobs),
            "queue_length": self.queue_manager.get_queue_length(),
            "active_workers": len([w for w in self.workers if w.status == WorkerStatus.BUSY]),
            "next_job_arrival_time": self.next_job_arrival_time,
            "total_jobs_created": self.total_jobs_created,
            "arrivals_remaining": len(self.arrival_schedule) - self.next_arrival_index
        }
    
    def run_simulation(self, duration: float = None, progress_callback=None) -> Dict[str, Any]:
        """
        Run simulation for specified duration or until termination criteria are met.
        
        Args:
            duration: Simulation duration in seconds (optional, uses termination_mode if not provided)
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary containing simulation results
        """
        self.running = True
        start_time = self.simulation_time
        
        # Determine termination criteria
        if duration is not None:
            # If duration is provided, use time-based termination
            target_time = start_time + duration
            termination_mode = 'time'
        else:
            # Use configured termination mode
            termination_mode = self.termination_mode
            target_time = float('inf')  # Will be overridden for time-based termination
            
            if termination_mode == 'time':
                # Use default duration if no duration provided
                duration = self.termination_config.get('duration', 3600.0)
                target_time = start_time + duration
        
        step_count = 0
        termination_reason = "unknown"
        
        print(f"Starting simulation with termination mode: {termination_mode}")
        if termination_mode == 'time':
            print(f"  Duration: {duration}, Target time: {target_time}")
        elif termination_mode == 'data_points':
            print(f"  Max data points: {self.max_data_points}")
        elif termination_mode == 'jobs':
            print(f"  Max jobs: {self.max_jobs}")
        elif termination_mode == 'completed_jobs':
            print(f"  Max completed jobs: {self.max_completed_jobs}")
        
        print(f"Initial next_job_arrival_time={self.next_job_arrival_time}")
        
        while self.running and self._should_continue_simulation(termination_mode, target_time):
            if not self.paused:  # Only step if not paused
                step_result = self.step_simulation()
                step_count += 1
                
                # Debug output for first few steps and periodically
                if step_count <= 10 or step_count % 100 == 0:
                    print(f"Step {step_count}: {step_result}")
                    if termination_mode == 'data_points':
                        print(f"  Data points logged: {self.data_points_logged}/{self.max_data_points}")
                    elif termination_mode == 'jobs':
                        print(f"  Jobs created: {self.total_jobs_created}/{self.max_jobs}")
                    elif termination_mode == 'completed_jobs':
                        completed = sum(w.completed_jobs for w in self.workers)
                        print(f"  Jobs completed: {completed}/{self.max_completed_jobs}")
                
                if progress_callback and step_count % 100 == 0:
                    progress = self._calculate_progress(termination_mode, start_time, target_time)
                    progress_callback(progress, step_result)
            else:
                # If paused, just wait a bit to avoid busy waiting
                import time
                time.sleep(0.01)
        
        # Determine termination reason
        if not self.running:
            termination_reason = "manual_stop"
        elif termination_mode == 'time' and self.simulation_time >= target_time:
            termination_reason = "time_limit_reached"
        elif termination_mode == 'data_points' and self.data_points_logged >= self.max_data_points:
            termination_reason = "data_points_limit_reached"
        elif termination_mode == 'jobs' and self.total_jobs_created >= self.max_jobs:
            termination_reason = "jobs_limit_reached"
        elif termination_mode == 'completed_jobs':
            completed = sum(w.completed_jobs for w in self.workers)
            if completed >= self.max_completed_jobs:
                termination_reason = "completed_jobs_limit_reached"
        
        # Final logging
        self.logger.log_simulation_state(
            self.simulation_time, self.queue_manager, 
            self.workers, self.total_jobs_created, force_flush=True
        )
        
        # Save all logs with final flush
        self.logger.save_json_logs(force_flush=True)
        
        return {
            "simulation_completed": True,
            "termination_reason": termination_reason,
            "termination_mode": termination_mode,
            "duration": self.simulation_time - start_time,
            "steps_executed": step_count,
            "data_points_logged": self.data_points_logged,
            "total_jobs_created": self.total_jobs_created,
            "total_jobs_completed": sum(w.completed_jobs for w in self.workers),
            "final_queue_length": self.queue_manager.get_queue_length(),
            "queue_stats": self.queue_manager.get_queue_stats(),
            "worker_stats": [w.get_stats() for w in self.workers],
            "log_summary": self.logger.get_log_summary()
        }
    
    def stop_simulation(self) -> None:
        """Stop the running simulation."""
        self.running = False
        self.paused = False
    
    def pause_simulation(self) -> None:
        """Pause the running simulation."""
        if self.running:
            self.paused = True
    
    def resume_simulation(self) -> None:
        """Resume the paused simulation."""
        if self.running:
            self.paused = False
    
    def step_once(self) -> Dict[str, Any]:
        """
        Execute a single simulation step while paused.
        
        Returns:
            Dictionary containing step results
        """
        if self.running and self.paused:
            return self.step_simulation()
        else:
            return {
                "error": "Can only step when simulation is running and paused",
                "running": self.running,
                "paused": self.paused
            }
    
    def step_time_period(self, time_period: float) -> Dict[str, Any]:
        """
        Execute simulation steps for a specific time period while paused.
        
        Args:
            time_period: Time period in seconds to step forward
            
        Returns:
            Dictionary containing step results
        """
        if not (self.running and self.paused):
            return {
                "error": "Can only step when simulation is running and paused",
                "running": self.running,
                "paused": self.paused
            }
        
        start_time = self.simulation_time
        target_time = start_time + time_period
        steps_executed = 0
        
        while self.simulation_time < target_time and self.running and self.paused:
            self.step_simulation()
            steps_executed += 1
        
        return {
            "simulation_time": self.simulation_time,
            "time_advanced": self.simulation_time - start_time,
            "steps_executed": steps_executed,
            "queue_length": self.queue_manager.get_queue_length(),
            "active_workers": len([w for w in self.workers if w.status == WorkerStatus.BUSY])
        }
    
    def _should_continue_simulation(self, termination_mode: str, target_time: float) -> bool:
        """
        Check if simulation should continue based on termination criteria.
        
        Args:
            termination_mode: The termination mode ('time', 'data_points', 'jobs', 'completed_jobs')
            target_time: Target time for time-based termination
            
        Returns:
            True if simulation should continue, False otherwise
        """
        if termination_mode == 'time':
            return self.simulation_time < target_time
        elif termination_mode == 'data_points':
            return self.data_points_logged < self.max_data_points
        elif termination_mode == 'jobs':
            return self.total_jobs_created < self.max_jobs
        elif termination_mode == 'completed_jobs':
            total_completed = sum(w.completed_jobs for w in self.workers)
            return total_completed < self.max_completed_jobs
        else:
            # Default to time-based if unknown mode
            return self.simulation_time < target_time
    
    def _calculate_progress(self, termination_mode: str, start_time: float, target_time: float) -> float:
        """
        Calculate simulation progress based on termination mode.
        
        Args:
            termination_mode: The termination mode
            start_time: Simulation start time
            target_time: Target time for time-based termination
            
        Returns:
            Progress as a float between 0.0 and 1.0
        """
        if termination_mode == 'time':
            if target_time == start_time:
                return 1.0
            return min(1.0, (self.simulation_time - start_time) / (target_time - start_time))
        elif termination_mode == 'data_points':
            if self.max_data_points == 0:
                return 1.0
            return min(1.0, self.data_points_logged / self.max_data_points)
        elif termination_mode == 'jobs':
            if self.max_jobs == 0:
                return 1.0
            return min(1.0, self.total_jobs_created / self.max_jobs)
        elif termination_mode == 'completed_jobs':
            if self.max_completed_jobs == 0:
                return 1.0
            total_completed = sum(w.completed_jobs for w in self.workers)
            return min(1.0, total_completed / self.max_completed_jobs)
        else:
            return 0.0
    
    def reset_simulation(self) -> None:
        """Reset simulation to initial state."""
        self.simulation_time = 0.0
        self.total_jobs_created = 0
        self.last_log_time = 0.0
        self.data_points_logged = 0
        self.running = False
        self.paused = False
        
        # Regenerate arrival schedule for scenario
        print(f"Regenerating arrival schedule for {self.scenario_duration_hours}-hour scenario...")
        self.arrival_schedule = self.job_factory.generate_scenario_arrivals(
            self.scenario_duration_seconds
        )
        self.next_arrival_index = 0
        print(f"Regenerated {len(self.arrival_schedule)} job arrivals for scenario")
        
        # Update next_job_arrival_time for compatibility
        self.next_job_arrival_time = (
            self.arrival_schedule[0] if self.arrival_schedule else float('inf')
        )
        
        # Reset components
        self.queue_manager = QueueManager()
        for worker in self.workers:
            worker.set_online()
        
        # Clear logs
        self.logger.clear_logs()
    
    def create_snapshot(self, description: str = None) -> SimulationSnapshot:
        """
        Create a snapshot of the current simulation state.
        
        Args:
            description: Optional description of the snapshot
            
        Returns:
            SimulationSnapshot object containing current state
        """
        # Serialize queue state
        queue_state = {
            "queue_length": self.queue_manager.get_queue_length(),
            "queue_stats": self.queue_manager.get_queue_stats(),
            "completed_jobs": [job.to_dict() for job in self.queue_manager.completed_jobs],
            "failed_jobs": [job.to_dict() for job in self.queue_manager.failed_jobs],
            "use_priority_queue": self.queue_manager.use_priority_queue,
            "max_queue_size": self.queue_manager.max_queue_size,
            "total_jobs_processed": self.queue_manager.total_jobs_processed
        }
        
        # Serialize workers state
        workers_state = [worker.to_dict() for worker in self.workers]
        
        # Add current jobs for each worker
        for i, worker in enumerate(self.workers):
            workers_state[i]["current_jobs_data"] = [job.to_dict() for job in worker.current_jobs]
        
        # Serialize factory state
        factory_state = self.job_factory.get_stats()
        factory_state["config"] = self._serialize_factory_config(self.job_factory.config)
        
        # Serialize logger state (just metadata, not full logs)
        logger_state = {
            "log_directory": str(self.logger.log_directory),
            "current_buffer_size": self.logger.current_buffer_size,
            "last_flush_time": self.logger.last_flush_time,
            "log_summary": self.logger.get_log_summary()
        }
        
        # Create simulation configuration snapshot
        simulation_config = {
            "time_step": self.time_step,
            "log_interval": self.log_interval,
            "num_workers": len(self.workers)
        }
        
        # Create snapshot
        snapshot = SimulationSnapshot(
            simulation_time=self.simulation_time,
            total_jobs_created=self.total_jobs_created,
            next_job_arrival_time=self.next_job_arrival_time,
            last_log_time=self.last_log_time,
            arrival_schedule=self.arrival_schedule.copy(),
            next_arrival_index=self.next_arrival_index,
            scenario_duration_hours=self.scenario_duration_hours,
            queue_state=queue_state,
            workers_state=workers_state,
            factory_state=factory_state,
            logger_state=logger_state,
            simulation_config=simulation_config,
            metadata={
                "description": description,
                "created_by": "SimulationEngine",
                "data_points_logged": self.data_points_logged,
                "termination_config": self.termination_config,
                "arrivals_remaining": len(self.arrival_schedule) - self.next_arrival_index
            }
        )
        
        return snapshot
    
    def _serialize_factory_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize factory configuration, converting enums to strings.
        
        Args:
            config: Original configuration dictionary
            
        Returns:
            Serialized configuration dictionary
        """
        serialized_config = {}
        
        for key, value in config.items():
            if key == "job_types" and isinstance(value, dict):
                # Convert JobType enum keys to strings
                serialized_job_types = {}
                for job_type, weight in value.items():
                    if hasattr(job_type, 'value'):
                        serialized_job_types[job_type.value] = weight
                    else:
                        serialized_job_types[str(job_type)] = weight
                serialized_config[key] = serialized_job_types
            elif isinstance(value, dict):
                # Recursively serialize nested dictionaries
                serialized_config[key] = self._serialize_factory_config(value)
            else:
                serialized_config[key] = value
        
        return serialized_config
    
    def _deserialize_factory_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deserialize factory configuration, converting strings back to enums where needed.
        
        Args:
            config: Serialized configuration dictionary
            
        Returns:
            Deserialized configuration dictionary
        """
        deserialized_config = {}
        
        for key, value in config.items():
            if key == "job_types" and isinstance(value, dict):
                # Convert string keys back to JobType enums
                deserialized_job_types = {}
                for job_type_str, weight in value.items():
                    try:
                        job_type = JobType(job_type_str)
                        deserialized_job_types[job_type] = weight
                    except ValueError:
                        # If conversion fails, keep as string
                        deserialized_job_types[job_type_str] = weight
                deserialized_config[key] = deserialized_job_types
            elif isinstance(value, dict):
                # Recursively deserialize nested dictionaries
                deserialized_config[key] = self._deserialize_factory_config(value)
            else:
                deserialized_config[key] = value
        
        return deserialized_config
    
    def save_snapshot(self, description: str = None, filename: str = None) -> str:
        """
        Save current simulation state to a snapshot file.
        
        Args:
            description: Optional description of the snapshot
            filename: Optional filename (auto-generated if None)
            
        Returns:
            Path to saved snapshot file
        """
        snapshot = self.create_snapshot(description)
        return self.snapshot_manager.save_snapshot(snapshot, filename)
    
    def restore_from_snapshot(self, snapshot_or_filename) -> Dict[str, Any]:
        """
        Restore simulation state from a snapshot.
        
        Args:
            snapshot_or_filename: SimulationSnapshot object or filename
            
        Returns:
            Dictionary containing restoration results
        """
        if self.running:
            return {"success": False, "error": "Cannot restore while simulation is running. Stop it first."}
        
        try:
            # Load snapshot if filename provided
            if isinstance(snapshot_or_filename, str):
                snapshot = self.snapshot_manager.load_snapshot(snapshot_or_filename)
            else:
                snapshot = snapshot_or_filename
            
            # Restore simulation state
            self.simulation_time = snapshot.simulation_time
            self.total_jobs_created = snapshot.total_jobs_created
            self.next_job_arrival_time = snapshot.next_job_arrival_time
            self.last_log_time = snapshot.last_log_time
            
            # Restore arrival schedule
            self.arrival_schedule = snapshot.arrival_schedule.copy() if snapshot.arrival_schedule else []
            self.next_arrival_index = snapshot.next_arrival_index
            self.scenario_duration_hours = snapshot.scenario_duration_hours
            self.scenario_duration_seconds = self.scenario_duration_hours * 3600
            
            # Restore queue state
            queue_state = snapshot.queue_state
            self.queue_manager = QueueManager(
                max_queue_size=queue_state.get("max_queue_size", 1000),
                use_priority_queue=queue_state.get("use_priority_queue", False)
            )
            self.queue_manager.total_jobs_processed = queue_state.get("total_jobs_processed", 0)
            
            # Restore completed and failed jobs
            for job_data in queue_state.get("completed_jobs", []):
                job = Job(
                    job_id=job_data["job_id"],
                    job_type=JobType(job_data["job_type"]),
                    size=job_data["size"],
                    duration=job_data["duration"],
                    priority=job_data["priority"]
                )
                job.status = JobStatus(job_data["status"])
                if job_data.get("created_at"):
                    job.created_at = datetime.fromisoformat(job_data["created_at"])
                if job_data.get("started_at"):
                    job.started_at = datetime.fromisoformat(job_data["started_at"])
                if job_data.get("completed_at"):
                    job.completed_at = datetime.fromisoformat(job_data["completed_at"])
                job.metadata = job_data.get("metadata", {})
                self.queue_manager.completed_jobs.append(job)
            
            for job_data in queue_state.get("failed_jobs", []):
                job = Job(
                    job_id=job_data["job_id"],
                    job_type=JobType(job_data["job_type"]),
                    size=job_data["size"],
                    duration=job_data["duration"],
                    priority=job_data["priority"]
                )
                job.status = JobStatus(job_data["status"])
                if job_data.get("created_at"):
                    job.created_at = datetime.fromisoformat(job_data["created_at"])
                if job_data.get("started_at"):
                    job.started_at = datetime.fromisoformat(job_data["started_at"])
                if job_data.get("completed_at"):
                    job.completed_at = datetime.fromisoformat(job_data["completed_at"])
                job.metadata = job_data.get("metadata", {})
                self.queue_manager.failed_jobs.append(job)
            
            # Restore workers
            workers_state = snapshot.workers_state
            self.workers = []
            
            for worker_data in workers_state:
                worker = Worker(
                    worker_id=worker_data["worker_id"],
                    processing_speed=worker_data["processing_speed"],
                    failure_rate=worker_data["failure_rate"],
                    efficiency_variance=worker_data["efficiency_variance"],
                    max_concurrent_jobs=worker_data["max_concurrent_jobs"]
                )
                
                # Restore worker stats
                worker.status = WorkerStatus(worker_data["status"])
                worker.completed_jobs = worker_data["completed_jobs"]
                worker.failed_jobs = worker_data["failed_jobs"]
                worker.total_processing_time = worker_data.get("total_processing_time", 0.0)
                worker.created_at = datetime.fromisoformat(worker_data["created_at"]) if worker_data.get("created_at") else datetime.now()
                if worker_data.get("last_job_completed_at"):
                    worker.last_job_completed_at = datetime.fromisoformat(worker_data["last_job_completed_at"])
                
                # Restore current jobs
                for job_data in worker_data.get("current_jobs_data", []):
                    job = Job(
                        job_id=job_data["job_id"],
                        job_type=JobType(job_data["job_type"]),
                        size=job_data["size"],
                        duration=job_data["duration"],
                        priority=job_data["priority"]
                    )
                    job.status = JobStatus(job_data["status"])
                    if job_data.get("created_at"):
                        job.created_at = datetime.fromisoformat(job_data["created_at"])
                    if job_data.get("started_at"):
                        job.started_at = datetime.fromisoformat(job_data["started_at"])
                    if job_data.get("completed_at"):
                        job.completed_at = datetime.fromisoformat(job_data["completed_at"])
                    job.metadata = job_data.get("metadata", {})
                    worker.current_jobs.append(job)
                
                self.workers.append(worker)
            
            # Restore factory state
            factory_state = snapshot.factory_state
            if "config" in factory_state:
                deserialized_config = self._deserialize_factory_config(factory_state["config"])
                self.job_factory.update_config(deserialized_config)
            if "jobs_created" in factory_state:
                self.job_factory.jobs_created = factory_state["jobs_created"]
            
            # Restore simulation configuration
            sim_config = snapshot.simulation_config
            self.time_step = sim_config.get("time_step", 0.1)
            self.log_interval = sim_config.get("log_interval", 1.0)
            
            return {
                "success": True,
                "snapshot_id": snapshot.snapshot_id,
                "restored_timestamp": snapshot.timestamp.isoformat(),
                "simulation_time": self.simulation_time,
                "total_jobs_created": self.total_jobs_created,
                "num_workers": len(self.workers),
                "queue_length": self.queue_manager.get_queue_length(),
                "metadata": snapshot.metadata
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to restore from snapshot: {e}",
                "error_type": type(e).__name__
            }
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """
        List available snapshots.
        
        Returns:
            List of snapshot information dictionaries
        """
        return self.snapshot_manager.list_snapshots()
    
    def delete_snapshot(self, filename: str) -> bool:
        """
        Delete a snapshot file.
        
        Args:
            filename: Snapshot filename
            
        Returns:
            True if deletion was successful, False otherwise
        """
        return self.snapshot_manager.delete_snapshot(filename)
