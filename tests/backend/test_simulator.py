"""
Unit tests for the backend simulator module.

Tests job generation, queue management, worker emulation, and statistical distributions.
"""

import pytest
import time
from datetime import datetime
from unittest.mock import patch, MagicMock
import numpy as np
from scipy import stats

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from simulator import (
    Job, JobType, JobStatus, 
    QueueManager, Worker, WorkerStatus,
    JobGenerator, DistributionType, WorkFactory
)


class TestJob:
    """Test cases for the Job class."""
    
    def test_job_creation(self):
        """Test basic job creation with default values."""
        job = Job()
        
        assert job.job_id is not None
        assert job.job_type == JobType.MIXED
        assert job.size == 1.0
        assert job.duration == 1.0
        assert job.priority == 0
        assert job.status == JobStatus.PENDING
        assert isinstance(job.created_at, datetime)
        assert job.started_at is None
        assert job.completed_at is None
        assert isinstance(job.metadata, dict)
    
    def test_job_creation_with_params(self):
        """Test job creation with custom parameters."""
        job = Job(
            job_type=JobType.MILP,
            size=5.0,
            duration=10.0,
            priority=3,
            metadata={"test": "data"}
        )
        
        assert job.job_type == JobType.MILP
        assert job.size == 5.0
        assert job.duration == 10.0
        assert job.priority == 3
        assert job.metadata["test"] == "data"
    
    def test_job_processing_workflow(self):
        """Test the complete job processing workflow."""
        job = Job()
        
        # Start processing
        job.start_processing()
        assert job.status == JobStatus.RUNNING
        assert job.started_at is not None
        
        # Complete processing
        job.complete_processing()
        assert job.status == JobStatus.COMPLETED
        assert job.completed_at is not None
        assert job.get_processing_time() is not None
    
    def test_job_failure(self):
        """Test job failure handling."""
        job = Job()
        job.start_processing()
        
        job.fail_processing("Test error")
        assert job.status == JobStatus.FAILED
        assert job.completed_at is not None
        assert "error" in job.metadata
        assert job.metadata["error"] == "Test error"
    
    def test_job_serialization(self):
        """Test job serialization to dictionary."""
        job = Job(job_type=JobType.HEURISTIC, size=2.5)
        job_dict = job.to_dict()
        
        assert job_dict["job_type"] == "heuristic"
        assert job_dict["size"] == 2.5
        assert job_dict["status"] == "pending"
        assert "job_id" in job_dict
        assert "created_at" in job_dict


class TestQueueManager:
    """Test cases for the QueueManager class."""
    
    def test_fifo_queue_creation(self):
        """Test FIFO queue manager creation."""
        qm = QueueManager(max_queue_size=10, use_priority_queue=False)
        
        assert qm.max_queue_size == 10
        assert qm.use_priority_queue == False
        assert qm.get_queue_length() == 0
        assert qm.is_empty() == True
        assert qm.is_full() == False
    
    def test_priority_queue_creation(self):
        """Test priority queue manager creation."""
        qm = QueueManager(max_queue_size=5, use_priority_queue=True)
        
        assert qm.use_priority_queue == True
        assert qm.max_queue_size == 5
    
    def test_fifo_job_ordering(self):
        """Test FIFO job ordering."""
        qm = QueueManager(use_priority_queue=False)
        
        job1 = Job(priority=1)
        job2 = Job(priority=5)
        job3 = Job(priority=3)
        
        qm.add_job(job1)
        qm.add_job(job2)
        qm.add_job(job3)
        
        # Should get jobs in FIFO order regardless of priority
        assert qm.get_next_job() == job1
        assert qm.get_next_job() == job2
        assert qm.get_next_job() == job3
    
    def test_priority_job_ordering(self):
        """Test priority-based job ordering."""
        qm = QueueManager(use_priority_queue=True)
        
        job1 = Job(priority=1)
        job2 = Job(priority=5) 
        job3 = Job(priority=3)
        
        qm.add_job(job1)
        qm.add_job(job2)
        qm.add_job(job3)
        
        # Should get highest priority job first
        next_job = qm.get_next_job()
        assert next_job.priority == 5
    
    def test_queue_capacity(self):
        """Test queue capacity limits."""
        qm = QueueManager(max_queue_size=2)
        
        job1 = Job()
        job2 = Job()
        job3 = Job()
        
        assert qm.add_job(job1) == True
        assert qm.add_job(job2) == True
        assert qm.is_full() == True
        assert qm.add_job(job3) == False  # Should fail, queue is full
    
    def test_job_completion_tracking(self):
        """Test job completion and failure tracking."""
        qm = QueueManager()
        job1 = Job()
        job2 = Job()
        
        qm.complete_job(job1)
        qm.fail_job(job2, "Test error")
        
        stats = qm.get_queue_stats()
        assert stats["completed_jobs"] == 1
        assert stats["failed_jobs"] == 1
        assert stats["total_jobs_processed"] == 2
        assert stats["success_rate"] == 0.5
    
    def test_peek_functionality(self):
        """Test peek functionality without removing jobs."""
        qm = QueueManager()
        job = Job()
        
        qm.add_job(job)
        peeked_job = qm.peek_next_job()
        
        assert peeked_job == job
        assert qm.get_queue_length() == 1  # Job should still be in queue


class TestWorker:
    """Test cases for the Worker class."""
    
    def test_worker_creation(self):
        """Test worker creation with default parameters."""
        worker = Worker()
        
        assert worker.worker_id is not None
        assert worker.processing_speed == 1.0
        assert worker.failure_rate == 0.01
        assert worker.efficiency_variance == 0.1
        assert worker.max_concurrent_jobs == 1
        assert worker.status == WorkerStatus.IDLE
        assert len(worker.current_jobs) == 0
    
    def test_worker_creation_with_params(self):
        """Test worker creation with custom parameters."""
        worker = Worker(
            worker_id="test-worker",
            processing_speed=2.0,
            failure_rate=0.05,
            efficiency_variance=0.2,
            max_concurrent_jobs=3
        )
        
        assert worker.worker_id == "test-worker"
        assert worker.processing_speed == 2.0
        assert worker.failure_rate == 0.05
        assert worker.efficiency_variance == 0.2
        assert worker.max_concurrent_jobs == 3
    
    def test_job_assignment(self):
        """Test job assignment to worker."""
        worker = Worker()
        job = Job(duration=1.0)  # Set a specific duration for testing
        
        assert worker.can_accept_job() == True
        assert worker.assign_job(job) == True
        assert worker.status == WorkerStatus.BUSY
        assert len(worker.current_jobs) == 1
        assert job.status == JobStatus.RUNNING
        
        # Test job processing - simulate time passing
        simulation_time = 0.0
        completed_jobs = worker.process_jobs(simulation_time)
        assert len(completed_jobs) == 0  # Job shouldn't be complete yet
        
        # Advance simulation time past job duration
        simulation_time = 10.0  # Well past the job duration
        completed_jobs = worker.process_jobs(simulation_time)
        assert len(completed_jobs) == 1  # Job should be completed now
        assert completed_jobs[0] == job
        assert worker.status == WorkerStatus.IDLE
    
    def test_concurrent_job_limit(self):
        """Test concurrent job limit enforcement."""
        worker = Worker(max_concurrent_jobs=2)
        
        job1 = Job()
        job2 = Job()
        job3 = Job()
        
        assert worker.assign_job(job1) == True
        assert worker.assign_job(job2) == True
        assert worker.assign_job(job3) == False  # Should fail, at capacity
    
    def test_worker_offline_online(self):
        """Test worker offline/online functionality."""
        worker = Worker()
        job = Job()
        
        worker.assign_job(job)
        interrupted_jobs = worker.set_offline()
        
        assert worker.status == WorkerStatus.OFFLINE
        assert len(interrupted_jobs) == 1
        assert interrupted_jobs[0] == job
        assert job.status == JobStatus.PENDING
        
        worker.set_online()
        assert worker.status == WorkerStatus.IDLE
    
    def test_worker_utilization(self):
        """Test worker utilization calculation."""
        worker = Worker(max_concurrent_jobs=4)
        
        assert worker.get_utilization() == 0.0
        
        worker.assign_job(Job())
        assert worker.get_utilization() == 0.25
        
        worker.assign_job(Job())
        assert worker.get_utilization() == 0.5
    
    def test_worker_stats(self):
        """Test worker statistics reporting."""
        worker = Worker(worker_id="test-worker")
        stats = worker.get_stats()
        
        assert stats["worker_id"] == "test-worker"
        assert stats["status"] == "idle"
        assert stats["completed_jobs"] == 0
        assert stats["failed_jobs"] == 0
        assert stats["utilization"] == 0.0
        assert "uptime_seconds" in stats


class TestJobGenerator:
    """Test cases for the JobGenerator class."""
    
    def test_normal_distribution(self):
        """Test normal distribution generation."""
        values = [
            JobGenerator.generate_value(
                DistributionType.NORMAL, mean=5.0, std=1.0
            ) for _ in range(1000)
        ]
        
        mean = np.mean(values)
        std = np.std(values)
        
        # Allow for some variance in the generated statistics
        assert 4.5 <= mean <= 5.5
        assert 0.8 <= std <= 1.2
        assert all(v > 0 for v in values)  # Should enforce minimum value
    
    def test_exponential_distribution(self):
        """Test exponential distribution generation."""
        lambda_param = 2.0
        values = [
            JobGenerator.generate_value(
                DistributionType.EXPONENTIAL, **{"lambda": lambda_param}
            ) for _ in range(1000)
        ]
        
        mean = np.mean(values)
        expected_mean = 1.0 / lambda_param
        
        # Exponential distribution mean should be 1/λ
        assert 0.4 <= mean <= 0.7  # Some tolerance for randomness
        assert all(v > 0 for v in values)
    
    def test_uniform_distribution(self):
        """Test uniform distribution generation."""
        low, high = 2.0, 8.0
        values = [
            JobGenerator.generate_value(
                DistributionType.UNIFORM, low=low, high=high
            ) for _ in range(1000)
        ]
        
        assert all(low <= v <= high for v in values)
        
        mean = np.mean(values)
        expected_mean = (low + high) / 2
        assert 4.5 <= mean <= 5.5  # Should be around 5.0
    
    def test_constant_distribution(self):
        """Test constant value generation."""
        constant_value = 42.0
        values = [
            JobGenerator.generate_value(
                DistributionType.CONSTANT, value=constant_value
            ) for _ in range(100)
        ]
        
        assert all(v == constant_value for v in values)
    
    def test_batch_generation(self):
        """Test batch value generation."""
        count = 50
        values = JobGenerator.generate_batch_values(
            DistributionType.NORMAL, count, mean=1.0, std=0.1
        )
        
        assert len(values) == count
        assert all(isinstance(v, (int, float)) for v in values)
    
    def test_distribution_stats(self):
        """Test distribution statistics calculation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = JobGenerator.get_distribution_stats(values)
        
        assert stats["count"] == 5
        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["median"] == 3.0
    
    def test_parameter_validation(self):
        """Test distribution parameter validation."""
        # Valid parameters
        is_valid, msg = JobGenerator.validate_distribution_params(
            DistributionType.NORMAL, {"mean": 1.0, "std": 0.5}
        )
        assert is_valid == True
        assert msg == ""
        
        # Invalid parameters (negative std)
        is_valid, msg = JobGenerator.validate_distribution_params(
            DistributionType.NORMAL, {"mean": 1.0, "std": -0.5}
        )
        assert is_valid == False
        assert "positive" in msg.lower()
    
    def test_distribution_info(self):
        """Test distribution information retrieval."""
        info = JobGenerator.get_distribution_info(DistributionType.NORMAL)
        
        assert "name" in info
        assert "parameters" in info
        assert "use_cases" in info
        assert "mean" in info["parameters"]
        assert "std" in info["parameters"]


class TestWorkFactory:
    """Test cases for the WorkFactory class."""
    
    def test_factory_creation(self):
        """Test work factory creation."""
        factory = WorkFactory()
        
        assert factory.config is not None
        assert factory.jobs_created == 0
        assert "job_size" in factory.config
        assert "job_duration" in factory.config
        assert "job_types" in factory.config
    
    def test_job_creation(self):
        """Test job creation through factory."""
        factory = WorkFactory()
        job = factory.create_job()
        
        assert isinstance(job, Job)
        assert job.size > 0
        assert job.duration > 0
        assert job.job_type in [JobType.MILP, JobType.HEURISTIC, JobType.ML, JobType.MIXED]
        assert factory.jobs_created == 1
    
    def test_job_creation_with_overrides(self):
        """Test job creation with parameter overrides."""
        factory = WorkFactory()
        override_params = {
            "size": 10.0,
            "duration": 5.0,
            "job_type": JobType.MILP
        }
        
        job = factory.create_job(override_params)
        
        assert job.size == 10.0
        assert job.duration == 5.0
        assert job.job_type == JobType.MILP
    
    def test_batch_creation(self):
        """Test batch job creation."""
        factory = WorkFactory()
        count = 10
        jobs = factory.create_batch(count)
        
        assert len(jobs) == count
        assert all(isinstance(job, Job) for job in jobs)
        assert factory.jobs_created == count
    
    def test_inter_arrival_time(self):
        """Test inter-arrival time generation."""
        factory = WorkFactory()
        
        arrival_times = [factory.generate_inter_arrival_time() for _ in range(100)]
        
        assert all(t > 0 for t in arrival_times)
        assert len(set(arrival_times)) > 1  # Should generate different values
    
    def test_config_update(self):
        """Test factory configuration updates."""
        factory = WorkFactory()
        original_config = factory.config.copy()
        
        new_config = {
            "job_size": {
                "distribution": DistributionType.CONSTANT,
                "params": {"value": 42.0}
            }
        }
        
        factory.update_config(new_config)
        
        assert factory.config["job_size"]["distribution"] == DistributionType.CONSTANT
        assert factory.config["job_size"]["params"]["value"] == 42.0
        
        # Other config should remain unchanged
        assert factory.config["job_duration"] == original_config["job_duration"]
    
    def test_factory_stats(self):
        """Test factory statistics."""
        factory = WorkFactory()
        factory.create_job()
        factory.create_job()
        
        stats = factory.get_stats()
        
        assert stats["jobs_created"] == 2
        assert "config" in stats


if __name__ == "__main__":
    pytest.main([__file__])
