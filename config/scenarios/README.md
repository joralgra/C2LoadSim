# Scenario Templates

This directory contains predefined scenario templates for different types of computational workloads in C2LoadSim.

## Available Templates

### 1. MILP (Mixed-Integer Linear Programming) - `milp.json`

**Purpose**: Simulates exact optimization algorithms like MILP solvers that require significant computational resources and have unpredictable execution times.

**Characteristics**:
- **Duration**: 2 hours (7200 seconds)
- **Job Arrival Rate**: Low (λ = 2 jobs/second) - Poisson process
- **Job Sizes**: Large, variable-sized jobs (Gamma distribution, k=10, θ=5)
- **Workers**: Few workers (3) with slow processing speed (μ=0.2, σ=0.05)
- **Resource Usage**: High CPU (80%), High Memory (1024MB average)

**Use Cases**:
- Testing autoscaling with resource-intensive workloads
- Validating system behavior under sustained high resource usage
- Training forecasting models for optimization workloads

---

### 2. Heuristics - `heuristics.json`

**Purpose**: Simulates fast heuristic and metaheuristic algorithms that process many small, predictable jobs quickly.

**Characteristics**:
- **Duration**: 1 hour (3600 seconds)
- **Job Arrival Rate**: High (λ = 10 jobs/second) - Poisson process
- **Job Sizes**: Small, consistent jobs (Normal distribution, μ=5, σ=1)
- **Workers**: Many workers (10) with fast processing speed (μ=0.9, σ=0.05)
- **Resource Usage**: Low CPU (30%), Low Memory (128MB average)

**Use Cases**:
- Testing high-throughput, low-latency scenarios
- Validating queue management with frequent job arrivals
- Simulating lightweight computational tasks

---

### 3. Machine Learning - `ml.json`

**Purpose**: Simulates machine learning workloads with irregular patterns, representing training jobs that may have varying computational requirements.

**Characteristics**:
- **Duration**: 3 hours (10800 seconds)
- **Job Arrival Rate**: Variable over time (NHPP, λ(t) = 5 + 3*sin(t/3600))
- **Job Sizes**: Highly variable (Lognormal distribution, μ=2, σ=1)
- **Workers**: Medium number (6) with erratic processing (μ=0.5, σ=0.2)
- **Resource Usage**: Medium CPU (60%), Medium Memory (512MB average)

**Use Cases**:
- Testing time-dependent workload patterns
- Simulating ML training pipelines with burst patterns
- Evaluating autoscaling responsiveness to load variations

---

### 4. Mixed Production Load - `mixed.json`

**Purpose**: Simulates a realistic production environment where multiple types of algorithms run concurrently, creating a heterogeneous workload.

**Characteristics**:
- **Duration**: 4 hours (14400 seconds)
- **Job Arrival Rate**: Variable (NHPP, λ(t) = 3 + 2*cos(t/1800))
- **Job Sizes**: Mixed distribution with three components:
  - 40% Gamma distribution (large MILP-like jobs)
  - 40% Normal distribution (heuristic-like jobs)
  - 20% Lognormal distribution (ML-like jobs)
- **Workers**: Balanced setup (8 workers) with Gamma-distributed processing speeds
- **Resource Usage**: Variable CPU (50%), Moderate Memory (256MB average)

**Use Cases**:
- Comprehensive testing of production-like scenarios
- Training forecasting models on realistic mixed workloads
- Evaluating autoscaling strategies under diverse job types

## Usage Instructions

### Loading a Scenario

1. **Via API**: Use the `/api/scenarios/load` endpoint with the scenario file
2. **Direct File**: Copy any template and modify parameters as needed
3. **Validation**: Always validate scenarios using `../validate_scenario.py` before use

### Customization Guidelines

- **Modify Duration**: Adjust `duration` field for shorter/longer simulations
- **Change Arrival Rates**: Modify `lambda` (Poisson) or `lambdaFunction` (NHPP)
- **Adjust Job Sizes**: Change distribution parameters in `jobSize`
- **Scale Workers**: Modify `workers.count` and associated parameters
- **Resource Limits**: Ensure CPU usage stays within [0.0, 1.0] bounds

### Validation

Before using any scenario (original or modified):

```bash
cd /path/to/config
python validate_scenario.py scenarios/milp.json
python validate_scenario.py scenarios/*.json  # Validate all
```

## Mathematical Expressions for NHPP

When using NHPP (Non-Homogeneous Poisson Process), the `lambdaFunction` can include:

- **Variables**: `t` (time in seconds)
- **Operations**: `+`, `-`, `*`, `/`, `(`, `)`
- **Functions**: `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`
- **Constants**: Any numeric values

**Examples**:
- `"5 + 3*sin(t/3600)"` - Sinusoidal pattern with 1-hour period
- `"2 + cos(t/1800)"` - Cosine pattern with 30-minute period  
- `"1 + 0.5*exp(-t/7200)"` - Exponential decay over 2 hours
