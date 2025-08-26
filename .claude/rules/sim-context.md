Here’s a full breakdown of how the simulator in your TFM works, according to the specification:

---

## **Simulator Purpose**

The simulator is designed to validate the behavior of the **deployment and autoscaling system** for algorithms in the C2NET platform.
It generates *snapshots of the system state* at each simulation period, allowing you to test:

* Exceptional scenarios
* Normal operation
* Load peaks

It also provides synthetic data for **forecasting models (ARIMA, LSTM, etc.)** and benchmarking.

---

## **Core Variables (per time step *t*)**

At each instant, the simulator produces:

* **Ep,i(t)**: Progress of execution of each active pod/worker ∈ \[0,1]
* **Npods(t)**: Number of pods executing jobs
* **Q(t)**: Queue length (pending jobs)
* **Cj,k(t)**: Relative size or computational cost of each queued job

  * Can be based on problem size (e.g., number of MILP variables) or estimated computational complexity.

---

## **Statistical Modeling**

The simulator uses probabilistic distributions to emulate realistic workload patterns:

* **Job arrivals:**

  * Initially normal distributions
  * Extended with **non-homogeneous Poisson processes (NHPP)** to represent time-dependent peaks.

* **Job sizes:**

  * Lognormal or Lognormal–Pareto mixtures (to capture heavy-tailed distributions and rare very large jobs).

* **Service speed (worker performance):**

  * Gamma distributions (low variance), modeling heterogeneous but stable worker execution speeds.

* **Progress dynamics:**

  * Truncated Normal distributions, varying by algorithm type.

---

## **Algorithm Profiles**

Each algorithm type has specific parameterizations:

* **MILP (exact algorithms):**

  * High variance in job sizes
  * Slow, irregular progress
* **Heuristics/metaheuristics:**

  * Small, short, predictable jobs
  * Steady progress
* **Machine Learning models:**

  * Jobs with lognormal sizes
  * Erratic progress and execution times

---

## **Defined Scenarios**

Four scenarios were implemented to capture different industrial realities:

1. **Exact Algorithms (MILP)**

   * **Arrivals:** Poisson homogeneous, constant λ
   * **Size:** Gamma, large jobs, low variance
   * **Workers:** Slow, unpredictable progress (low mean Normal)

2. **Heuristics**

   * **Arrivals:** Poisson homogeneous, higher λ
   * **Size:** Normal with small mean, low variance
   * **Workers:** Fast and regular progress

3. **Machine Learning**

   * **Arrivals:** NHPP with time-dependent λ(t), simulating load bursts (e.g., training at night)
   * **Size:** Lognormal (short vs. long iterations)
   * **Workers:** Erratic progress (high variance Normal)

4. **Mixed Production Load**

   * **Arrivals:** Combination of Poisson + NHPP
   * **Size:** Mixed distribution (Gamma for MILP, Normal for heuristics, Lognormal for ML)
   * **Workers:** Heterogeneous profiles (slow + fast)
   * **Goal:** Emulates real industrial settings with concurrent heterogeneous workloads.

---

## **Architecture of the Simulation**

* **WorkFactory:** Generates jobs based on chosen distributions.
* **Global Queue:** Stores pending jobs.
* **Workers:** Consume jobs, process them, and update progress.

This structure allows comparison between:

* Purely reactive autoscaling (Kubernetes HPA)
* Proactive autoscaling (forecast-based with ARIMA/LSTM).

---

## **Strategic Role**

The simulator is not just for stress-testing:

* It **provides training data** for forecasting models.
* It supports **comparisons of autoscaling strategies**.
* It enables **repeatable validation** without depending on real production traces.

---

Would you like me to also **draw a conceptual diagram** of the simulator architecture (WorkFactory → Queue → Workers) with variables (Ep,i, Npods, Q, Cj,k) to help visualize it?
