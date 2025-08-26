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
from pathlib import Path


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
            "job_type": self.job_type.value,
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
    CONSTANT = "constant"


class JobGenerator:
    """
    Generates random values based on statistical distributions using scipy.stats.
    
    Provides methods to generate job attributes (size, duration, inter-arrival times)
    using various probability distributions with enhanced statistical capabilities.
    """
    
    @staticmethod
    def generate_value(distribution: DistributionType, **params) -> float:
        """
        Generate a random value based on the specified distribution using scipy.stats.
        
        Args:
            distribution: Type of distribution to use
            **params: Distribution-specific parameters
            
        Returns:
            Generated random value
        """
        if distribution == DistributionType.NORMAL:
            mu = params.get('mean', 1.0)
            sigma = params.get('std', 0.1)
            return max(0.01, stats.norm.rvs(loc=mu, scale=sigma))
        
        elif distribution == DistributionType.LOGNORMAL:
            # For lognormal: if X ~ LogNormal(μ, σ²), then ln(X) ~ Normal(μ, σ²)
            mu = params.get('mean_log', 0.0)
            sigma = params.get('std_log', 0.5)
            return stats.lognorm.rvs(s=sigma, scale=np.exp(mu))
        
        elif distribution == DistributionType.EXPONENTIAL:
            # scipy uses scale parameter (1/rate), not rate parameter
            rate = params.get('lambda', 1.0)
            scale = 1.0 / rate
            return stats.expon.rvs(scale=scale)
        
        elif distribution == DistributionType.UNIFORM:
            low = params.get('low', 0.0)
            high = params.get('high', 2.0)
            return stats.uniform.rvs(loc=low, scale=high-low)
        
        elif distribution == DistributionType.GAMMA:
            # scipy gamma uses shape (a) and scale parameters
            alpha = params.get('alpha', 2.0)  # shape parameter
            beta = params.get('beta', 1.0)    # scale parameter
            return stats.gamma.rvs(a=alpha, scale=beta)
        
        elif distribution == DistributionType.POISSON:
            mu = params.get('mu', 1.0)
            # For continuous approximation, use normal distribution
            return max(0.01, stats.norm.rvs(loc=mu, scale=np.sqrt(mu)))
        
        elif distribution == DistributionType.CONSTANT:
            return params.get('value', 1.0)
        
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
    
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
    def generate_batch_values(distribution: DistributionType, count: int, **params) -> List[float]:
        """
        Generate a batch of random values from the same distribution.
        
        Args:
            distribution: Type of distribution to use
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
    def validate_distribution_params(distribution: DistributionType, params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate parameters for a given distribution.
        
        Args:
            distribution: Type of distribution
            params: Parameters to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if distribution == DistributionType.NORMAL:
                mean = params.get('mean', 1.0)
                std = params.get('std', 0.1)
                if std <= 0:
                    return False, "Standard deviation must be positive"
                    
            elif distribution == DistributionType.LOGNORMAL:
                std_log = params.get('std_log', 0.5)
                if std_log <= 0:
                    return False, "Log standard deviation must be positive"
                    
            elif distribution == DistributionType.EXPONENTIAL:
                lambd = params.get('lambda', 1.0)
                if lambd <= 0:
                    return False, "Lambda (rate) parameter must be positive"
                    
            elif distribution == DistributionType.UNIFORM:
                low = params.get('low', 0.0)
                high = params.get('high', 2.0)
                if low >= high:
                    return False, "High bound must be greater than low bound"
                    
            elif distribution == DistributionType.GAMMA:
                alpha = params.get('alpha', 2.0)
                beta = params.get('beta', 1.0)
                if alpha <= 0 or beta <= 0:
                    return False, "Alpha and beta parameters must be positive"
                    
            elif distribution == DistributionType.POISSON:
                mu = params.get('mu', 1.0)
                if mu <= 0:
                    return False, "Mu parameter must be positive"
            
            # Test generation to ensure parameters work
            JobGenerator.generate_value(distribution, **params)
            return True, ""
            
        except Exception as e:
            return False, f"Parameter validation failed: {str(e)}"
    
    @staticmethod
    def create_distribution_config(
        distribution: DistributionType, 
        **params
    ) -> Dict[str, Any]:
        """
        Create a standardized distribution configuration.
        
        Args:
            distribution: Type of distribution
            **params: Distribution parameters
            
        Returns:
            Standardized configuration dictionary
        """
        is_valid, error_msg = JobGenerator.validate_distribution_params(distribution, params)
        if not is_valid:
            raise ValueError(f"Invalid distribution parameters: {error_msg}")
        
        return {
            "distribution": distribution,
            "params": params,
            "info": JobGenerator.get_distribution_info(distribution)
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
                "distribution": DistributionType.LOGNORMAL,
                "params": {"mean_log": 0.0, "std_log": 0.5}
            },
            "job_duration": {
                "distribution": DistributionType.GAMMA,
                "params": {"alpha": 2.0, "beta": 1.0}
            },
            "job_priority": {
                "distribution": DistributionType.UNIFORM,
                "params": {"low": 0, "high": 10}
            },
            "inter_arrival_time": {
                "distribution": DistributionType.EXPONENTIAL,
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
    
    def generate_inter_arrival_time(self) -> float:
        """
        Generate the time until the next job arrives.
        
        Returns:
            Time in seconds until next job arrival
        """
        config = self.config["inter_arrival_time"]
        return self.generator.generate_value(
            config["distribution"], 
            **config["params"]
        )
    
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
            return override_params["job_type"]
        
        # Weighted random selection
        job_types = list(self.config["job_types"].keys())
        weights = list(self.config["job_types"].values())
        
        rand_val = random.random()
        cumulative = 0.0
        
        for job_type, weight in zip(job_types, weights):
            cumulative += weight
            if rand_val <= cumulative:
                return job_type
        
        # Fallback to first type
        return job_types[0]
    
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
    worker utilization, and queue statistics over time.
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
        
        # Initialize log structures
        self.json_logs: List[Dict[str, Any]] = []
        self.csv_headers_written = False
        
        # Clear existing logs
        self._initialize_log_files()
    
    def _initialize_log_files(self) -> None:
        """Initialize log files with headers."""
        # Initialize JSON log
        self.json_logs = []
        
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
    
    def log_simulation_state(
        self, 
        simulation_time: float,
        queue_manager: QueueManager,
        workers: List[Worker],
        total_jobs_created: int = 0
    ) -> None:
        """
        Log the current simulation state.
        
        Args:
            simulation_time: Current simulation time
            queue_manager: Queue manager instance
            workers: List of worker instances
            total_jobs_created: Total number of jobs created so far
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
        
        # Create log entry
        log_entry = {
            "timestamp": timestamp,
            "simulation_time": simulation_time,
            "total_jobs_created": total_jobs_created,
            "queue_stats": queue_stats,
            "worker_stats": {
                "total_workers": len(workers),
                "active_workers": active_workers,
                "average_utilization": avg_utilization,
                "individual_utilizations": worker_utilizations
            },
            "performance_metrics": {
                "throughput": throughput,
                "total_completed": total_completed,
                "total_failed": total_failed,
                "success_rate": total_completed / (total_completed + total_failed) if (total_completed + total_failed) > 0 else 0.0
            }
        }
        
        # Add to JSON log
        self.json_logs.append(log_entry)
        
        # Write to CSV
        csv_row = [
            timestamp, simulation_time, total_jobs_created, queue_stats["queue_length"],
            queue_stats["completed_jobs"], queue_stats["failed_jobs"], len(workers),
            active_workers, queue_stats["queue_length"], avg_utilization, throughput
        ]
        
        with open(self.csv_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_row)
    
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
            timestamp, job.job_id, job.job_type.value, job.size, job.duration,
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
    
    def save_json_logs(self) -> None:
        """Save all JSON logs to file."""
        with open(self.json_log_file, 'w') as f:
            json.dump(self.json_logs, f, indent=2)
    
    def get_log_summary(self) -> Dict[str, Any]:
        """
        Get a summary of logged data.
        
        Returns:
            Dictionary containing log summary statistics
        """
        if not self.json_logs:
            return {"message": "No logs available"}
        
        total_entries = len(self.json_logs)
        latest_entry = self.json_logs[-1]
        
        return {
            "total_log_entries": total_entries,
            "simulation_duration": latest_entry["simulation_time"],
            "final_queue_length": latest_entry["queue_stats"]["queue_length"],
            "total_jobs_processed": latest_entry["queue_stats"]["total_jobs_processed"],
            "final_success_rate": latest_entry["performance_metrics"]["success_rate"],
            "average_throughput": latest_entry["performance_metrics"]["throughput"],
            "log_files": {
                "json_log": str(self.json_log_file),
                "csv_log": str(self.csv_log_file),
                "jobs_log": str(self.jobs_csv_file),
                "workers_log": str(self.workers_csv_file)
            }
        }
    
    def clear_logs(self) -> None:
        """Clear all log data and reinitialize files."""
        self._initialize_log_files()
        print(f"Logs cleared and reinitialized in {self.log_directory}")


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
        log_directory: str = "logs"
    ):
        """
        Initialize the simulation engine.
        
        Args:
            num_workers: Number of worker instances to create
            queue_config: Configuration for queue manager
            worker_config: Configuration for worker creation
            factory_config: Configuration for job factory
            log_directory: Directory for log files
        """
        # Initialize components
        self.queue_manager = QueueManager(**(queue_config or {}))
        self.workers = [
            Worker(**(worker_config or {})) for _ in range(num_workers)
        ]
        self.job_factory = WorkFactory(factory_config)
        self.logger = SimulationLogger(log_directory)
        
        # Simulation state
        self.simulation_time = 0.0
        self.running = False
        self.paused = False
        self.total_jobs_created = 0
        self.next_job_arrival_time = 0.0
        
        # Configuration
        self.time_step = 0.1  # Simulation time step in seconds
        self.log_interval = 1.0  # How often to log state
        self.last_log_time = 0.0
    
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
        # Generate new jobs based on arrival patterns
        while self.simulation_time >= self.next_job_arrival_time:
            self.create_job()
            inter_arrival_time = self.job_factory.generate_inter_arrival_time()
            self.next_job_arrival_time += inter_arrival_time
        
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
            self.last_log_time = self.simulation_time
        
        # Advance time
        self.simulation_time += self.time_step
        
        return {
            "simulation_time": self.simulation_time,
            "jobs_assigned": len(assigned_jobs),
            "jobs_completed": len(completed_jobs),
            "queue_length": self.queue_manager.get_queue_length(),
            "active_workers": len([w for w in self.workers if w.status == WorkerStatus.BUSY])
        }
    
    def run_simulation(self, duration: float, progress_callback=None) -> Dict[str, Any]:
        """
        Run simulation for specified duration.
        
        Args:
            duration: Simulation duration in seconds
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary containing simulation results
        """
        self.running = True
        start_time = self.simulation_time
        target_time = start_time + duration
        
        step_count = 0
        
        while self.running and self.simulation_time < target_time:
            if not self.paused:  # Only step if not paused
                step_result = self.step_simulation()
                step_count += 1
                
                if progress_callback and step_count % 100 == 0:
                    progress = (self.simulation_time - start_time) / duration
                    progress_callback(progress, step_result)
            else:
                # If paused, just wait a bit to avoid busy waiting
                import time
                time.sleep(0.01)
        
        # Final logging
        self.logger.log_simulation_state(
            self.simulation_time, self.queue_manager, 
            self.workers, self.total_jobs_created
        )
        
        # Save all logs
        self.logger.save_json_logs()
        
        return {
            "simulation_completed": True,
            "duration": self.simulation_time - start_time,
            "steps_executed": step_count,
            "total_jobs_created": self.total_jobs_created,
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
    
    def reset_simulation(self) -> None:
        """Reset simulation to initial state."""
        self.simulation_time = 0.0
        self.total_jobs_created = 0
        self.next_job_arrival_time = 0.0
        self.last_log_time = 0.0
        self.running = False
        self.paused = False
        
        # Reset components
        self.queue_manager = QueueManager()
        for worker in self.workers:
            worker.set_online()
        
        # Clear logs
        self.logger.clear_logs()
