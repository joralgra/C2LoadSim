#!/usr/bin/env python3
"""
Scenario validation script for C2LoadSim.

This script validates JSON scenario files against the defined schema.
"""

import json
import sys
import argparse
from pathlib import Path
from jsonschema import validate, ValidationError, Draft7Validator
from typing import List, Tuple


def load_schema() -> dict:
    """Load the JSON schema from schema.json file."""
    schema_path = Path(__file__).parent / "schema.json"
    try:
        with open(schema_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Schema file not found at {schema_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in schema file: {e}")
        sys.exit(1)


def validate_scenario_data(scenario_data: dict, schema: dict) -> Tuple[bool, List[str]]:
    """
    Validate scenario data directly against the schema.
    
    Args:
        scenario_data: Dictionary containing scenario data
        schema: JSON schema dictionary
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Validate against schema
    validator = Draft7Validator(schema)
    validation_errors = list(validator.iter_errors(scenario_data))
    
    if validation_errors:
        for error in validation_errors:
            error_path = " -> ".join(str(p) for p in error.absolute_path)
            if error_path:
                errors.append(f"Error at '{error_path}': {error.message}")
            else:
                errors.append(f"Error: {error.message}")
        return False, errors
    
    # Additional custom validations
    custom_errors = _custom_validations(scenario_data)
    if custom_errors:
        return False, custom_errors
        
    return True, []


def validate_scenario_file(scenario_path: Path, schema: dict) -> Tuple[bool, List[str]]:
    """
    Validate a single scenario file against the schema.
    
    Args:
        scenario_path: Path to the scenario JSON file
        schema: JSON schema dictionary
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    try:
        with open(scenario_path, 'r') as f:
            scenario_data = json.load(f)
    except FileNotFoundError:
        return False, [f"File not found: {scenario_path}"]
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON in {scenario_path}: {e}"]
    
    return validate_scenario_data(scenario_data, schema)


def _custom_validations(scenario_data: dict) -> List[str]:
    """
    Perform additional custom validations beyond the schema.
    
    Args:
        scenario_data: Loaded scenario JSON data
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    # Validate mixture components weights sum to ~1.0
    if scenario_data.get("jobSize", {}).get("type") == "mixture":
        components = scenario_data["jobSize"]["components"]
        total_weight = sum(comp["weight"] for comp in components)
        if not (0.99 <= total_weight <= 1.01):  # Allow small floating point errors
            errors.append(f"Mixture component weights sum to {total_weight}, should sum to 1.0")
    
    # Validate NHPP lambda function syntax (basic check)
    if scenario_data.get("arrival", {}).get("type") == "nhpp":
        lambda_func = scenario_data["arrival"]["lambdaFunction"]
        # Basic syntax validation - check for common mathematical functions
        allowed_funcs = ["sin", "cos", "tan", "exp", "log", "sqrt", "abs"]
        allowed_chars = set("0123456789+-*/().,t ")
        for func in allowed_funcs:
            allowed_chars.update(set(func))
        
        if not all(c in allowed_chars for c in lambda_func):
            errors.append(f"NHPP lambda function contains invalid characters: {lambda_func}")
    
    # Validate CPU usage bounds
    cpu_config = scenario_data.get("workers", {}).get("cpuUsage", {})
    if cpu_config.get("type") == "normal":
        mu = cpu_config.get("mu", 0)
        sigma = cpu_config.get("sigma", 0)
        if mu + 3*sigma > 1.0:  # Check if 99.7% of values could exceed 1.0
            errors.append(f"CPU usage distribution (mu={mu}, sigma={sigma}) may generate values > 1.0")
        if mu - 3*sigma < 0.0:  # Check if 99.7% of values could be negative
            errors.append(f"CPU usage distribution (mu={mu}, sigma={sigma}) may generate negative values")
    
    return errors


def main():
    """Main function to validate scenario files."""
    parser = argparse.ArgumentParser(description="Validate C2LoadSim scenario files")
    parser.add_argument("files", nargs="+", help="Scenario JSON files to validate")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Show detailed validation information")
    
    args = parser.parse_args()
    
    # Load schema
    schema = load_schema()
    
    # Validate each file
    all_valid = True
    for file_path in args.files:
        scenario_path = Path(file_path)
        
        if args.verbose:
            print(f"Validating {scenario_path}...")
        
        is_valid, error_messages = validate_scenario_file(scenario_path, schema)
        
        if is_valid:
            print(f"✓ {scenario_path} is valid")
        else:
            print(f"✗ {scenario_path} is invalid:")
            for error in error_messages:
                print(f"  - {error}")
            all_valid = False
    
    if all_valid:
        print(f"\nAll {len(args.files)} scenario files are valid!")
        sys.exit(0)
    else:
        print(f"\nValidation failed for some files.")
        sys.exit(1)


if __name__ == "__main__":
    main()
