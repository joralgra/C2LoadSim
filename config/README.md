# Scenario Configuration Schema

This directory contains the JSON schema for C2LoadSim simulation scenarios and predefined scenario templates.

## Schema Definition

The `schema.json` file defines the structure and validation rules for simulation scenario files. All scenario JSON files must conform to this schema.

### Required Fields

- **scenarioName**: Unique identifier for the scenario
- **duration**: Simulation duration in seconds
- **seed**: Random seed for reproducibility
- **arrival**: Job arrival pattern configuration
- **jobSize**: Job size distribution
- **workers**: Worker/pod configuration

### Arrival Patterns

Two types of arrival patterns are supported:

1. **Poisson**: Homogeneous Poisson process with constant λ
   ```json
   {
     "type": "poisson",
     "lambda": 5.0
   }
   ```

2. **NHPP**: Non-homogeneous Poisson process with time-dependent λ(t)
   ```json
   {
     "type": "nhpp",
     "lambdaFunction": "5 + 3*sin(t/3600)"
   }
   ```

### Job Size Distributions

Supported distributions:

1. **Normal**: `{"type": "normal", "parameters": {"mu": 10, "sigma": 2}}`
2. **Lognormal**: `{"type": "lognormal", "parameters": {"mu": 2, "sigma": 1}}`
3. **Gamma**: `{"type": "gamma", "parameters": {"k": 2, "theta": 5}}`
4. **Mixture**: Combination of multiple distributions with weights

### Worker Configuration

Workers are configured with:
- **count**: Number of workers/pods
- **speedDistribution**: Processing speed distribution (normal or gamma)
- **cpuUsage**: CPU utilization distribution (normal, 0.0-1.0)
- **memoryUsage**: Memory usage distribution (normal, in MB)

## Scenario Templates

The `scenarios/` directory contains predefined templates:

- `milp.json`: Long-running, resource-intensive optimization problems
- `heuristics.json`: Fast, small heuristic algorithms
- `ml.json`: Machine learning workloads with irregular patterns
- `mixed.json`: Combined workload with multiple algorithm types

## Validation

Use the `validate_scenario.py` script to validate scenario files against the schema before use.
