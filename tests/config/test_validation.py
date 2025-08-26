#!/usr/bin/env python3
"""
Test suite for scenario validation functionality.

Tests the JSON schema validation, custom validations, and error handling
for C2LoadSim scenario files.
"""

import json
import pytest
import tempfile
from pathlib import Path
import sys
import os

# Add the config directory to Python path for imports
config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config')
sys.path.insert(0, config_path)

from validate_scenario import load_schema, validate_scenario_data, validate_scenario_file


class TestSchemaValidation:
    """Test basic JSON schema validation."""
    
    @pytest.fixture
    def schema(self):
        """Load the JSON schema for testing."""
        return load_schema()
    
    @pytest.fixture
    def valid_scenario(self):
        """A valid scenario for testing."""
        return {
            "scenarioName": "TestScenario",
            "duration": 3600,
            "seed": 42,
            "arrival": {
                "type": "poisson",
                "lambda": 5
            },
            "jobSize": {
                "type": "normal",
                "parameters": {
                    "mu": 10,
                    "sigma": 2
                }
            },
            "workers": {
                "count": 5,
                "speedDistribution": {
                    "type": "normal",
                    "mu": 0.8,
                    "sigma": 0.1
                },
                "cpuUsage": {
                    "type": "normal",
                    "mu": 0.5,
                    "sigma": 0.1
                },
                "memoryUsage": {
                    "type": "normal",
                    "mu": 256,
                    "sigma": 64
                }
            }
        }
    
    def test_valid_scenario(self, schema, valid_scenario):
        """Test that a valid scenario passes validation."""
        is_valid, errors = validate_scenario_data(valid_scenario, schema)
        assert is_valid, f"Valid scenario failed validation: {errors}"
        assert len(errors) == 0
    
    def test_missing_required_fields(self, schema):
        """Test validation fails when required fields are missing."""
        # Missing scenarioName
        scenario = {
            "duration": 3600,
            "seed": 42,
            "arrival": {"type": "poisson", "lambda": 5},
            "jobSize": {"type": "normal", "parameters": {"mu": 10, "sigma": 2}},
            "workers": {
                "count": 5,
                "speedDistribution": {"type": "normal", "mu": 0.8, "sigma": 0.1},
                "cpuUsage": {"type": "normal", "mu": 0.5, "sigma": 0.1},
                "memoryUsage": {"type": "normal", "mu": 256, "sigma": 64}
            }
        }
        
        is_valid, errors = validate_scenario_data(scenario, schema)
        assert not is_valid
        assert any("scenarioName" in error for error in errors)
    
    def test_invalid_duration(self, schema, valid_scenario):
        """Test validation fails for invalid duration values."""
        # Negative duration
        scenario = valid_scenario.copy()
        scenario["duration"] = -100
        
        is_valid, errors = validate_scenario_data(scenario, schema)
        assert not is_valid
        assert any("minimum" in error.lower() for error in errors)
    
    def test_invalid_arrival_type(self, schema, valid_scenario):
        """Test validation fails for invalid arrival types."""
        scenario = valid_scenario.copy()
        scenario["arrival"] = {"type": "invalid_type", "lambda": 5}
        
        is_valid, errors = validate_scenario_data(scenario, schema)
        assert not is_valid
    
    def test_poisson_missing_lambda(self, schema, valid_scenario):
        """Test validation fails when Poisson arrival is missing lambda."""
        scenario = valid_scenario.copy()
        scenario["arrival"] = {"type": "poisson"}
        
        is_valid, errors = validate_scenario_data(scenario, schema)
        assert not is_valid
        assert any("arrival" in error and ("valid" in error or "schema" in error) for error in errors)
    
    def test_nhpp_missing_lambda_function(self, schema, valid_scenario):
        """Test validation fails when NHPP arrival is missing lambdaFunction."""
        scenario = valid_scenario.copy()
        scenario["arrival"] = {"type": "nhpp"}
        
        is_valid, errors = validate_scenario_data(scenario, schema)
        assert not is_valid
        assert any("arrival" in error and ("valid" in error or "schema" in error) for error in errors)
    
    def test_invalid_job_size_distribution(self, schema, valid_scenario):
        """Test validation fails for invalid job size distributions."""
        scenario = valid_scenario.copy()
        scenario["jobSize"] = {"type": "invalid_dist", "parameters": {"mu": 10}}
        
        is_valid, errors = validate_scenario_data(scenario, schema)
        assert not is_valid
    
    def test_missing_distribution_parameters(self, schema, valid_scenario):
        """Test validation fails when distribution parameters are missing."""
        scenario = valid_scenario.copy()
        scenario["jobSize"] = {"type": "normal", "parameters": {"mu": 10}}  # Missing sigma
        
        is_valid, errors = validate_scenario_data(scenario, schema)
        assert not is_valid
        assert any("jobSize" in error and ("valid" in error or "schema" in error) for error in errors)
    
    def test_invalid_worker_count(self, schema, valid_scenario):
        """Test validation fails for invalid worker count."""
        scenario = valid_scenario.copy()
        scenario["workers"]["count"] = 0  # Should be minimum 1
        
        is_valid, errors = validate_scenario_data(scenario, schema)
        assert not is_valid
        assert any("minimum" in error.lower() for error in errors)
    
    def test_negative_distribution_parameters(self, schema, valid_scenario):
        """Test validation fails for negative standard deviations."""
        scenario = valid_scenario.copy()
        scenario["jobSize"]["parameters"]["sigma"] = -1
        
        is_valid, errors = validate_scenario_data(scenario, schema)
        assert not is_valid
        assert any("jobSize" in error and ("valid" in error or "schema" in error) for error in errors)


class TestCustomValidations:
    """Test custom validation logic beyond the JSON schema."""
    
    @pytest.fixture
    def schema(self):
        return load_schema()
    
    def test_mixture_weights_sum_validation(self, schema):
        """Test that mixture component weights are validated to sum to 1.0."""
        scenario = {
            "scenarioName": "MixtureTest",
            "duration": 3600,
            "seed": 42,
            "arrival": {"type": "poisson", "lambda": 5},
            "jobSize": {
                "type": "mixture",
                "components": [
                    {"weight": 0.3, "type": "normal", "parameters": {"mu": 10, "sigma": 2}},
                    {"weight": 0.3, "type": "gamma", "parameters": {"k": 2, "theta": 5}}
                    # Weights sum to 0.6, not 1.0
                ]
            },
            "workers": {
                "count": 5,
                "speedDistribution": {"type": "normal", "mu": 0.8, "sigma": 0.1},
                "cpuUsage": {"type": "normal", "mu": 0.5, "sigma": 0.1},
                "memoryUsage": {"type": "normal", "mu": 256, "sigma": 64}
            }
        }
        
        is_valid, errors = validate_scenario_data(scenario, schema)
        assert not is_valid
        assert any("weight" in error.lower() and "sum" in error.lower() for error in errors)
    
    def test_mixture_weights_valid_sum(self, schema):
        """Test that mixture with correct weights passes validation."""
        scenario = {
            "scenarioName": "MixtureTest",
            "duration": 3600,
            "seed": 42,
            "arrival": {"type": "poisson", "lambda": 5},
            "jobSize": {
                "type": "mixture",
                "components": [
                    {"weight": 0.6, "type": "normal", "parameters": {"mu": 10, "sigma": 2}},
                    {"weight": 0.4, "type": "gamma", "parameters": {"k": 2, "theta": 5}}
                ]
            },
            "workers": {
                "count": 5,
                "speedDistribution": {"type": "normal", "mu": 0.8, "sigma": 0.1},
                "cpuUsage": {"type": "normal", "mu": 0.5, "sigma": 0.1},
                "memoryUsage": {"type": "normal", "mu": 256, "sigma": 64}
            }
        }
        
        is_valid, errors = validate_scenario_data(scenario, schema)
        assert is_valid, f"Mixture with correct weights failed: {errors}"
    
    def test_cpu_usage_bounds_validation(self, schema):
        """Test validation of CPU usage distribution bounds."""
        scenario = {
            "scenarioName": "CPUTest",
            "duration": 3600,
            "seed": 42,
            "arrival": {"type": "poisson", "lambda": 5},
            "jobSize": {"type": "normal", "parameters": {"mu": 10, "sigma": 2}},
            "workers": {
                "count": 5,
                "speedDistribution": {"type": "normal", "mu": 0.8, "sigma": 0.1},
                "cpuUsage": {"type": "normal", "mu": 0.9, "sigma": 0.5},  # May exceed 1.0
                "memoryUsage": {"type": "normal", "mu": 256, "sigma": 64}
            }
        }
        
        is_valid, errors = validate_scenario_data(scenario, schema)
        assert not is_valid
        assert any("cpu" in error.lower() and ("1.0" in error or "values" in error) for error in errors)
    
    def test_nhpp_lambda_function_validation(self, schema):
        """Test validation of NHPP lambda function syntax."""
        scenario = {
            "scenarioName": "NHPPTest",
            "duration": 3600,
            "seed": 42,
            "arrival": {"type": "nhpp", "lambdaFunction": "5 + invalid_func(t)"},
            "jobSize": {"type": "normal", "parameters": {"mu": 10, "sigma": 2}},
            "workers": {
                "count": 5,
                "speedDistribution": {"type": "normal", "mu": 0.8, "sigma": 0.1},
                "cpuUsage": {"type": "normal", "mu": 0.5, "sigma": 0.1},
                "memoryUsage": {"type": "normal", "mu": 256, "sigma": 64}
            }
        }
        
        is_valid, errors = validate_scenario_data(scenario, schema)
        assert not is_valid
        assert any("lambda function" in error.lower() or "invalid" in error.lower() for error in errors)


class TestFileValidation:
    """Test file-based validation functions."""
    
    @pytest.fixture
    def schema(self):
        return load_schema()
    
    def test_valid_file_validation(self, schema):
        """Test validation of a valid scenario file."""
        valid_scenario = {
            "scenarioName": "FileTest",
            "duration": 3600,
            "seed": 42,
            "arrival": {"type": "poisson", "lambda": 5},
            "jobSize": {"type": "normal", "parameters": {"mu": 10, "sigma": 2}},
            "workers": {
                "count": 5,
                "speedDistribution": {"type": "normal", "mu": 0.8, "sigma": 0.1},
                "cpuUsage": {"type": "normal", "mu": 0.5, "sigma": 0.1},
                "memoryUsage": {"type": "normal", "mu": 256, "sigma": 64}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(valid_scenario, f, indent=2)
            temp_path = Path(f.name)
        
        try:
            is_valid, errors = validate_scenario_file(temp_path, schema)
            assert is_valid, f"Valid file failed validation: {errors}"
            assert len(errors) == 0
        finally:
            temp_path.unlink()
    
    def test_invalid_json_file(self, schema):
        """Test validation of file with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json content")
            temp_path = Path(f.name)
        
        try:
            is_valid, errors = validate_scenario_file(temp_path, schema)
            assert not is_valid
            assert any("json" in error.lower() for error in errors)
        finally:
            temp_path.unlink()
    
    def test_missing_file(self, schema):
        """Test validation of non-existent file."""
        non_existent = Path("/tmp/non_existent_file.json")
        
        is_valid, errors = validate_scenario_file(non_existent, schema)
        assert not is_valid
        assert any("not found" in error.lower() for error in errors)


class TestIntegrationScenarios:
    """Test validation of complete scenario templates."""
    
    @pytest.fixture
    def schema(self):
        return load_schema()
    
    def test_all_predefined_scenarios(self, schema):
        """Test that all predefined scenario templates are valid."""
        scenarios_dir = Path(__file__).parent.parent / "config" / "scenarios"
        
        if not scenarios_dir.exists():
            pytest.skip("Scenarios directory not found")
        
        for scenario_file in scenarios_dir.glob("*.json"):
            is_valid, errors = validate_scenario_file(scenario_file, schema)
            assert is_valid, f"Predefined scenario {scenario_file.name} failed validation: {errors}"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
