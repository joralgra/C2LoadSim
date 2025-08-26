# Product Requirements Document (PRD)  
**Feature:** Computation Simulator for Synthetic Load Generation  
**Location:** `/tasks/prd-computation-simulator.md`

---

## 1. Introduction / Overview
The computation simulator is a tool designed to generate *synthetic computational load data* for training forecasting models (e.g., ARIMA, LSTM) and for validating scaling strategies in the C2NET platform.  

It creates configurable job streams and emulates the execution of tasks by pods/workers under different scenarios. The simulator’s outputs can be stored as structured logs (JSON/CSV) and visualized in real time via a web-based interface.  

This addresses the need for reproducible, synthetic, and controllable datasets in environments where real workload traces are either unavailable or insufficient.

---

## 2. Goals
1. Generate synthetic computational load data using statistical distributions.  
2. Support scenario-based configurations (MILP, heuristics, ML, mixed) via JSON input files.  
3. Emulate pod/worker task processing to approximate real cloud execution patterns.  
4. Provide structured logs (JSON/CSV) for analysis and forecasting training.  
5. Include system-level metrics (CPU and memory usage per worker).  
6. Offer a React-based frontend to visualize job arrivals, queue lengths, and worker activity in real time.  
7. Allow pausing/freezing simulations, exporting snapshot states, and resuming later.  

---

## 3. User Stories
- **As a researcher**, I want to generate synthetic computational load so I can train forecasting models without needing real traces.  
- **As a developer**, I want to configure scenarios via JSON files, so I can easily test different system behaviors without modifying code.  
- **As a student**, I want to visualize the simulator in real time through a web interface, so I can understand how pods, jobs, and queues interact.  
- **As an analyst**, I want to export simulation logs in JSON/CSV, so I can perform offline statistical analysis and forecasting model validation.  
- **As an operator**, I want each worker’s CPU and memory usage logged, so I can evaluate realistic resource consumption patterns.  

---

## 4. Functional Requirements
1. **Scenario Configuration**
   - Accept scenario definition files in JSON format.  
   - Each JSON must specify:  
     - Job arrival distribution (e.g., Poisson, NHPP).  
     - Job size distribution (Normal, Lognormal, Gamma, Mixture).  
     - Worker processing speed distribution.  
     - CPU and memory usage distribution per worker.  
     - Scenario duration and seed for reproducibility.  

2. **Scenario Execution**
   - Run one scenario at a time (batch execution not required).  
   - Generate jobs according to the chosen scenario.  
   - Workers (pods) emulate processing tasks until completion.  
   - Progress of each worker follows defined statistical parameters.  

3. **Output Logging**
   - Export results as JSON and CSV.  
   - Logs must include:  
     - Timestamp `t`  
     - Number of active workers `Npods(t)`  
     - Queue length `Q(t)`  
     - Job details (size, duration, type, progress)  
     - Worker states, including CPU and memory usage  

4. **Real-Time Visualization**
   - Provide a React-based web interface.  
   - Display queue length, worker utilization, CPU/memory, and progress bars.  
   - Allow pause/play controls and stepping through simulation time windows.  

5. **Simulation Snapshots**
   - Freezing a simulation must store the full system state in a file.  
   - Simulator must allow resuming from a saved snapshot file.  

6. **Scenario Library**
   - Provide predefined JSON scenario templates corresponding to TFM specifications:  
     - **MILP**: long-running, high-variance tasks.  
     - **Heuristics**: frequent, small tasks with low variance.  
     - **Machine Learning**: irregular jobs with lognormal size distributions.  
     - **Mixed**: combination of the above, with probabilistic selection.  

---

## 5. Non-Goals (Out of Scope)
- Running real workloads from production systems.  
- Frontend JSON editor (scenarios are provided externally).  
- Batch execution of multiple scenarios in sequence.  
- Predictive modeling (simulator only generates synthetic data, does not forecast).  
- Extreme edge cases (e.g., zero jobs or massive burst loads) are not prioritized for the MVP.  

---

## 6. Design Considerations
- **Backend:** Python module responsible for generating jobs, managing queues, emulating workers, and logging results.  
- **Frontend:** React web application for real-time visualization of simulation state.  
- **Data Storage:** Logs saved locally as JSON/CSV; snapshot files for paused simulations.  
- **Configuration:** JSON scenario files loaded via backend API endpoint.  
- **Reproducibility:** Support for seeding random distributions.  

---

## 7. Technical Considerations
- Must support at least the following distributions: Normal, Truncated Normal, Lognormal, Gamma, Poisson (homogeneous and non-homogeneous).  
- Modular design to allow adding new distributions/scenarios.  
- Configurable simulation step granularity (e.g., 1s, 100ms).  
- Backend–frontend communication via WebSocket for real-time updates.  
- Containerized deployment (Docker) for reproducibility and portability.  

---

## 8. Success Metrics
- Simulator reproduces the 4 predefined TFM scenarios.  
- JSON/CSV logs contain complete state (jobs, queues, workers, CPU, memory).  
- Snapshots can be saved and reloaded successfully.  
- Web interface displays system state accurately in real time.  
- Researchers use generated datasets to train ARIMA/LSTM models with <15% mean error on validation.  

---

## 9. Open Questions (Updated)
1. Should the simulator allow **scaling workers dynamically during runtime** (mimicking auto-scaling), or keep worker count fixed per scenario?  
2. Should the web interface support **multiple concurrent viewers** of the same simulation?  
3. Should the simulator include a **“fast-forward” mode** (run simulation faster than real time) for quick testing of long scenarios?  
4. Should CPU/memory usage be modeled with simple distributions, or correlated with job size/duration for realism?  

---

## Appendix A: JSON Schema for Scenario Files

### JSON Schema (simplified)
```json
{
  "scenarioName": "string",
  "duration": 3600,
  "seed": 123,
  "arrival": {
    "type": "poisson" | "nhpp",
    "lambda": 5,
    "lambdaFunction": "0.5*t" 
  },
  "jobSize": {
    "type": "normal" | "lognormal" | "gamma" | "mixture",
    "parameters": { "mu": 10, "sigma": 2 }
  },
  "workers": {
    "count": 5,
    "speedDistribution": { "type": "gamma", "k": 2, "theta": 1 },
    "cpuUsage": { "type": "normal", "mu": 0.5, "sigma": 0.1 },
    "memoryUsage": { "type": "normal", "mu": 256, "sigma": 64 }
  }
}
