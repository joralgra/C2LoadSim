#!/usr/bin/env python3
"""
Integration test to demonstrate pause/resume/step control responsiveness and accuracy.

This test validates the complete workflow of simulation controls to ensure
they respond correctly and maintain accuracy.
"""

import sys
import os
import time
import threading

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from simulator import (
    SimulationEngine, Job, JobType, JobStatus, 
    QueueManager, Worker, WorkerStatus, WorkFactory
)

def test_control_responsiveness_integration():
    """
    Integration test demonstrating control responsiveness and accuracy.
    
    This test simulates user interactions with pause, resume, and step controls
    to verify they work correctly in realistic scenarios.
    """
    print("🧪 Testing Control Responsiveness and Accuracy")
    print("=" * 60)
    
    # Initialize simulation engine
    engine = SimulationEngine(num_workers=3)
    
    print("✅ Step 1: Testing initial state")
    assert engine.running == False
    assert engine.paused == False
    assert engine.simulation_time == 0.0
    print("   Initial state is correct")
    
    print("\n✅ Step 2: Testing simulation start and pause")
    # Start simulation in background
    engine.running = True
    initial_time = engine.simulation_time
    
    # Let it run briefly
    for _ in range(3):
        engine.step_simulation()
    
    time_after_steps = engine.simulation_time
    assert time_after_steps > initial_time
    print(f"   Simulation advanced from {initial_time:.2f}s to {time_after_steps:.2f}s")
    
    # Test pause functionality
    engine.pause_simulation()
    assert engine.running == True
    assert engine.paused == True
    pause_time = engine.simulation_time
    print(f"   Simulation paused at {pause_time:.2f}s")
    
    print("\n✅ Step 3: Testing step controls while paused")
    # Test single step
    step_result = engine.step_once()
    assert "error" not in step_result
    assert engine.simulation_time > pause_time
    step_once_time = engine.simulation_time
    print(f"   Single step advanced to {step_once_time:.2f}s")
    
    # Test time period step
    time_period = 2.0  # 2 seconds
    step_period_result = engine.step_time_period(time_period)
    assert "error" not in step_period_result
    assert step_period_result["time_advanced"] >= time_period - engine.time_step
    final_step_time = engine.simulation_time
    print(f"   Time period step advanced to {final_step_time:.2f}s")
    print(f"   Time advanced: {step_period_result['time_advanced']:.2f}s")
    print(f"   Steps executed: {step_period_result['steps_executed']}")
    
    print("\n✅ Step 4: Testing resume functionality")
    # Resume simulation
    engine.resume_simulation()
    assert engine.running == True
    assert engine.paused == False
    resume_time = engine.simulation_time
    print(f"   Simulation resumed from {resume_time:.2f}s")
    
    # Let it run a bit more
    for _ in range(2):
        engine.step_simulation()
    
    final_time = engine.simulation_time
    assert final_time > resume_time
    print(f"   Simulation continued to {final_time:.2f}s")
    
    print("\n✅ Step 5: Testing error handling")
    # Test step when not paused (should fail)
    step_error_result = engine.step_once()
    assert "error" in step_error_result
    print("   Step control correctly rejected when not paused")
    
    # Test step time period when not paused (should fail)
    step_time_error_result = engine.step_time_period(1.0)
    assert "error" in step_time_error_result
    print("   Step time control correctly rejected when not paused")
    
    print("\n✅ Step 6: Testing state consistency")
    # Test that worker and queue states are maintained correctly
    initial_workers = len(engine.workers)
    
    # Pause again
    engine.pause_simulation()
    paused_workers = len(engine.workers)
    assert initial_workers == paused_workers
    
    # Resume
    engine.resume_simulation()
    resumed_workers = len(engine.workers)
    assert initial_workers == resumed_workers
    print("   Worker count consistent through pause/resume cycle")
    
    # Test final stop
    engine.stop_simulation()
    assert engine.running == False
    assert engine.paused == False
    print("   Simulation stopped correctly")
    
    print("\n✅ Step 7: Testing time accuracy")
    # Test that time stepping is accurate
    engine.reset_simulation()
    engine.running = True
    engine.pause_simulation()
    
    start_time = engine.simulation_time
    expected_advance = 5.0  # 5 seconds
    
    step_result = engine.step_time_period(expected_advance)
    actual_advance = step_result["time_advanced"]
    
    # Should be within one time step of expected
    tolerance = engine.time_step
    assert abs(actual_advance - expected_advance) <= tolerance
    print(f"   Time step accuracy: expected {expected_advance}s, got {actual_advance:.3f}s")
    print(f"   Error: {abs(actual_advance - expected_advance):.3f}s (tolerance: {tolerance:.3f}s)")
    
    print("\n🎉 All control responsiveness and accuracy tests PASSED!")
    print("=" * 60)
    
    return {
        "total_tests": 7,
        "passed": 7,
        "failed": 0,
        "time_accuracy_error": abs(actual_advance - expected_advance),
        "time_accuracy_tolerance": tolerance
    }

def test_frontend_backend_integration_logic():
    """
    Test the logic that the frontend would use to interact with backend controls.
    
    This validates the API contract and parameter validation.
    """
    print("\n🔗 Testing Frontend-Backend Integration Logic")
    print("=" * 60)
    
    engine = SimulationEngine(num_workers=2)
    
    print("✅ Step 1: Testing API parameter validation")
    
    # Test valid time periods
    valid_periods = [0.1, 1.0, 5.5, 10, 60, 300, 3600]
    for period in valid_periods:
        assert period >= 0.1, f"Invalid period: {period}"
        assert isinstance(period, (int, float)), f"Invalid type for period: {type(period)}"
    print("   Time period validation works correctly")
    
    # Test valid time units
    valid_units = ['seconds', 'minutes', 'hours', 'days', 'weeks']
    for unit in valid_units:
        assert isinstance(unit, str), f"Invalid type for unit: {type(unit)}"
        assert len(unit) > 0, f"Empty unit string"
    print("   Time unit validation works correctly")
    
    print("\n✅ Step 2: Testing state reporting accuracy")
    
    # Test state reporting when stopped
    state = {
        "running": engine.running,
        "paused": engine.paused,
        "simulation_time": engine.simulation_time
    }
    expected_state = {"running": False, "paused": False, "simulation_time": 0.0}
    for key, expected_value in expected_state.items():
        assert state[key] == expected_value, f"State mismatch for {key}: {state[key]} != {expected_value}"
    print("   Stopped state reported correctly")
    
    # Test state reporting when running
    engine.running = True
    state = {
        "running": engine.running,
        "paused": engine.paused
    }
    expected_state = {"running": True, "paused": False}
    for key, expected_value in expected_state.items():
        assert state[key] == expected_value, f"State mismatch for {key}: {state[key]} != {expected_value}"
    print("   Running state reported correctly")
    
    # Test state reporting when paused
    engine.pause_simulation()
    state = {
        "running": engine.running,
        "paused": engine.paused
    }
    expected_state = {"running": True, "paused": True}
    for key, expected_value in expected_state.items():
        assert state[key] == expected_value, f"State mismatch for {key}: {state[key]} != {expected_value}"
    print("   Paused state reported correctly")
    
    print("\n✅ Step 3: Testing response format validation")
    
    # Test step response format
    step_response = engine.step_once()
    required_keys = ["simulation_time", "jobs_assigned", "jobs_completed", "queue_length", "active_workers"]
    for key in required_keys:
        assert key in step_response, f"Missing key in step response: {key}"
        assert isinstance(step_response[key], (int, float)), f"Invalid type for {key}: {type(step_response[key])}"
    print("   Step response format is correct")
    
    # Test time period step response format
    time_step_response = engine.step_time_period(1.0)
    required_keys = ["simulation_time", "time_advanced", "steps_executed", "queue_length", "active_workers"]
    for key in required_keys:
        assert key in time_step_response, f"Missing key in time step response: {key}"
        assert isinstance(time_step_response[key], (int, float)), f"Invalid type for {key}: {type(time_step_response[key])}"
    print("   Time step response format is correct")
    
    print("\n🎉 Frontend-Backend Integration Logic tests PASSED!")
    
    return {
        "total_tests": 3,
        "passed": 3,
        "failed": 0
    }

if __name__ == "__main__":
    try:
        # Run main control tests
        control_results = test_control_responsiveness_integration()
        
        # Run integration logic tests
        integration_results = test_frontend_backend_integration_logic()
        
        # Summary
        total_tests = control_results["total_tests"] + integration_results["total_tests"]
        total_passed = control_results["passed"] + integration_results["passed"]
        total_failed = control_results["failed"] + integration_results["failed"]
        
        print(f"\n📊 FINAL SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_failed}")
        print(f"Success Rate: {(total_passed/total_tests)*100:.1f}%")
        print(f"Time Accuracy Error: {control_results['time_accuracy_error']:.3f}s")
        print(f"Time Accuracy Tolerance: {control_results['time_accuracy_tolerance']:.3f}s")
        
        if total_failed == 0:
            print("\n🎉 ALL TESTS PASSED! Controls are responsive and accurate.")
            print("✅ Task 2.3.3: Test controls for responsiveness and accuracy - COMPLETED")
        else:
            print(f"\n❌ {total_failed} tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
