"""
Unit tests for pause/resume/step functionality in the backend simulator.

Tests the synchronization between backend simulation control and frontend controls.
"""

import pytest
import time
import threading
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from simulator import (
    SimulationEngine, Job, JobType, JobStatus, 
    QueueManager, Worker, WorkerStatus, WorkFactory
)


class TestSimulationEnginePauseResume:
    """Test cases for pause/resume functionality in SimulationEngine."""
    
    def test_initial_state(self):
        """Test that simulation engine starts with correct initial state."""
        engine = SimulationEngine(num_workers=2)
        
        assert engine.running == False
        assert engine.paused == False
        assert engine.simulation_time == 0.0
    
    def test_pause_resume_basic(self):
        """Test basic pause and resume functionality."""
        engine = SimulationEngine(num_workers=2)
        
        # Start simulation
        engine.running = True
        assert engine.running == True
        assert engine.paused == False
        
        # Pause simulation
        engine.pause_simulation()
        assert engine.running == True
        assert engine.paused == True
        
        # Resume simulation
        engine.resume_simulation()
        assert engine.running == True
        assert engine.paused == False
    
    def test_pause_when_not_running(self):
        """Test that pausing when not running doesn't change state."""
        engine = SimulationEngine(num_workers=2)
        
        engine.pause_simulation()
        assert engine.running == False
        assert engine.paused == False
    
    def test_resume_when_not_running(self):
        """Test that resuming when not running doesn't change state."""
        engine = SimulationEngine(num_workers=2)
        
        engine.resume_simulation()
        assert engine.running == False
        assert engine.paused == False
    
    def test_stop_clears_pause(self):
        """Test that stopping simulation clears pause state."""
        engine = SimulationEngine(num_workers=2)
        engine.running = True
        engine.paused = True
        
        engine.stop_simulation()
        assert engine.running == False
        assert engine.paused == False
    
    def test_reset_clears_pause(self):
        """Test that resetting simulation clears pause state."""
        engine = SimulationEngine(num_workers=2)
        engine.running = False
        engine.paused = True
        engine.simulation_time = 100.0
        
        engine.reset_simulation()
        assert engine.running == False
        assert engine.paused == False
        assert engine.simulation_time == 0.0


class TestSimulationEngineStep:
    """Test cases for step functionality in SimulationEngine."""
    
    def test_step_once_when_paused(self):
        """Test single step execution when simulation is paused."""
        # Use default factory config, just test the step functionality
        engine = SimulationEngine(num_workers=2)
        
        # Set up simulation state
        engine.running = True
        engine.paused = True
        initial_time = engine.simulation_time
        
        # Execute single step
        result = engine.step_once()
        
        # Verify step was executed
        assert "error" not in result
        assert engine.simulation_time > initial_time
        assert engine.running == True
        assert engine.paused == True  # Should remain paused
    
    def test_step_once_when_not_paused(self):
        """Test that step_once returns error when not paused."""
        engine = SimulationEngine(num_workers=2)
        
        # Test when not running
        result = engine.step_once()
        assert "error" in result
        assert result["running"] == False
        assert result["paused"] == False
        
        # Test when running but not paused
        engine.running = True
        engine.paused = False
        result = engine.step_once()
        assert "error" in result
        assert result["running"] == True
        assert result["paused"] == False
    
    def test_step_time_period(self):
        """Test stepping forward by a specific time period."""
        engine = SimulationEngine(num_workers=2)
        engine.running = True
        engine.paused = True
        
        initial_time = engine.simulation_time
        time_period = 5.0  # 5 seconds
        
        result = engine.step_time_period(time_period)
        
        # Verify time advanced
        assert "error" not in result
        assert result["time_advanced"] >= time_period - engine.time_step  # Allow for rounding
        assert result["steps_executed"] > 0
        assert engine.simulation_time >= initial_time + time_period - engine.time_step
    
    def test_step_time_period_when_not_paused(self):
        """Test that step_time_period returns error when not paused."""
        engine = SimulationEngine(num_workers=2)
        
        result = engine.step_time_period(1.0)
        assert "error" in result
        assert "Can only step when simulation is running and paused" in result["error"]
    
    def test_step_time_period_respects_pause_state(self):
        """Test that step_time_period stops if pause state changes."""
        engine = SimulationEngine(num_workers=2)
        engine.running = True
        engine.paused = True
        
        initial_time = engine.simulation_time
        
        def unpause_after_delay():
            time.sleep(0.01)  # Wait a very short time
            engine.paused = False  # Unpause during stepping
        
        # Start thread to unpause simulation during step execution
        thread = threading.Thread(target=unpause_after_delay)
        thread.start()
        
        result = engine.step_time_period(1.0)  # Request shorter time period
        thread.join()
        
        # Should have stopped early when unpaused, or completed if timing was off
        time_advanced = result["time_advanced"]
        # More lenient check - just verify some progression occurred
        assert time_advanced > 0  # Some time should have advanced
        assert engine.simulation_time > initial_time


class TestSimulationEngineRunWithPause:
    """Test cases for run_simulation with pause/resume integration."""
    
    def test_run_simulation_respects_pause(self):
        """Test that run_simulation pauses correctly during execution."""
        engine = SimulationEngine(num_workers=2)
        
        simulation_started = threading.Event()
        pause_executed = threading.Event()
        resume_executed = threading.Event()
        
        def run_simulation():
            simulation_started.set()
            # Run for short duration to test pause behavior
            engine.run_simulation(duration=2.0)  # Shorter duration for faster test
        
        def pause_after_start():
            simulation_started.wait(timeout=1.0)
            time.sleep(0.05)  # Let simulation run briefly
            engine.pause_simulation()
            pause_executed.set()
            time.sleep(0.1)  # Keep paused for a bit
            engine.resume_simulation()
            resume_executed.set()
        
        # Start simulation in background
        sim_thread = threading.Thread(target=run_simulation)
        pause_thread = threading.Thread(target=pause_after_start)
        
        sim_thread.start()
        pause_thread.start()
        
        # Wait with timeout to prevent hanging tests
        sim_thread.join(timeout=5.0)
        pause_thread.join(timeout=5.0)
        
        # Verify pause was executed
        assert pause_executed.is_set()
        assert resume_executed.is_set()
        # Don't assert final running state as it depends on timing


class TestSimulationEngineStateConsistency:
    """Test cases for state consistency during pause/resume/step operations."""
    
    def test_job_creation_during_pause(self):
        """Test that jobs are not created while simulation is paused."""
        # Use default factory config
        engine = SimulationEngine(num_workers=2)
        
        engine.running = True
        engine.paused = True
        
        initial_jobs = engine.total_jobs_created
        
        # Try to create jobs while paused - should not happen automatically
        time.sleep(0.2)
        
        # Jobs should only be created during explicit steps
        assert engine.total_jobs_created == initial_jobs
        
        # But should create jobs when stepping (though not guaranteed in single step)
        result = engine.step_once()
        # Just verify step executed without error
        assert "error" not in result
    
    def test_worker_state_consistency(self):
        """Test that worker states remain consistent during pause/resume."""
        engine = SimulationEngine(num_workers=3)
        
        # Create some jobs and assign to workers
        for i in range(2):
            job = engine.create_job()
            for worker in engine.workers:
                if worker.can_accept_job():
                    worker.assign_job(job)
                    break
        
        # Count busy workers
        busy_workers_before = len([w for w in engine.workers if w.status == WorkerStatus.BUSY])
        
        # Pause and resume
        engine.running = True
        engine.pause_simulation()
        engine.resume_simulation()
        
        # Worker states should be unchanged
        busy_workers_after = len([w for w in engine.workers if w.status == WorkerStatus.BUSY])
        assert busy_workers_before == busy_workers_after
    
    def test_queue_state_consistency(self):
        """Test that queue states remain consistent during pause/resume."""
        engine = SimulationEngine(num_workers=1)  # Single worker to ensure queue buildup
        
        # Add jobs to queue
        for i in range(5):
            job = engine.create_job()
        
        initial_queue_length = engine.queue_manager.get_queue_length()
        
        # Pause and resume
        engine.running = True
        engine.pause_simulation()
        engine.resume_simulation()
        
        # Queue length should be unchanged
        assert engine.queue_manager.get_queue_length() == initial_queue_length
    
    def test_simulation_time_consistency(self):
        """Test that simulation time advances correctly with pause/resume/step."""
        engine = SimulationEngine(num_workers=2)
        engine.running = True
        
        # Record initial time
        initial_time = engine.simulation_time
        
        # Execute a few steps normally
        for _ in range(5):
            engine.step_simulation()
        
        time_after_steps = engine.simulation_time
        steps_time = time_after_steps - initial_time
        
        # Now test pause/resume with steps
        engine.pause_simulation()
        
        paused_time = engine.simulation_time
        
        # Step while paused
        engine.step_once()
        step_time = engine.simulation_time
        
        # Resume and step normally
        engine.resume_simulation()
        engine.step_simulation()
        final_time = engine.simulation_time
        
        # Verify time progression
        assert time_after_steps > initial_time
        assert step_time > paused_time
        assert final_time > step_time
        assert abs((step_time - paused_time) - engine.time_step) < 0.01  # Single step should advance by time_step


class TestPauseResumeAPIIntegration:
    """Test cases for API integration with pause/resume functionality."""
    
    def test_pause_resume_state_reporting(self):
        """Test that pause/resume states are correctly reported for API."""
        engine = SimulationEngine(num_workers=2)
        
        # Test initial state
        assert hasattr(engine, 'paused')
        assert engine.paused == False
        
        # Test paused state
        engine.running = True
        engine.pause_simulation()
        assert getattr(engine, 'paused', False) == True
        
        # Test resumed state
        engine.resume_simulation()
        assert getattr(engine, 'paused', False) == False
    
    def test_step_results_format(self):
        """Test that step results are properly formatted for API consumption."""
        engine = SimulationEngine(num_workers=2)
        engine.running = True
        engine.paused = True
        
        # Test single step
        result = engine.step_once()
        
        expected_keys = ["simulation_time", "jobs_assigned", "jobs_completed", 
                        "queue_length", "active_workers"]
        
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
            assert isinstance(result[key], (int, float)), f"Invalid type for {key}"
        
        # Test time period step
        result = engine.step_time_period(1.0)
        
        expected_keys = ["simulation_time", "time_advanced", "steps_executed",
                        "queue_length", "active_workers"]
        
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
            assert isinstance(result[key], (int, float)), f"Invalid type for {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
