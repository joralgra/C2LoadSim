## Relevant Files

- `backend/simulator.py` - Core backend module for generating jobs, managing queues, and emulating workers.
- `frontend/src/components/SimulatorDashboard.jsx` - React component for real-time visualization of the simulation state.
- `frontend/src/api/simulatorAPI.js` - API handler for backend communication.
- `config/scenarios/` - Directory for predefined JSON scenario templates.
- `config/schema.json` - JSON Schema definition for validating scenario files.
- `config/validate_scenario.py` - Python script for validating scenario files against the schema.
- `config/README.md` - Documentation for the scenario configuration system.
- `config/scenarios/README.md` - Documentation for predefined scenario templates.
- `config/scenarios/milp.json` - MILP (Mixed-Integer Linear Programming) scenario template.
- `config/scenarios/heuristics.json` - Heuristics algorithms scenario template.
- `config/scenarios/ml.json` - Machine Learning workloads scenario template.
- `config/scenarios/mixed.json` - Mixed production load scenario template.
- `tests/config/test_validation.py` - Unit tests for scenario validation functionality.
- `tests/backend/test_simulator.py` - Unit tests for the backend simulator module.
- `tests/frontend/test_SimulatorDashboard.jsx` - Unit tests for the React dashboard component.
- `tests/backend/test_pause_resume_step.py` - Comprehensive unit tests for pause/resume/step functionality in the backend simulator.
- `tests/integration/test_control_responsiveness.py` - Integration test demonstrating control responsiveness and accuracy for pause/resume/step controls.
- `frontend/src/components/SimulatorDashboard.test.tsx` - Frontend unit tests for SimulatorDashboard component controls.
- `frontend/src/setupTests.ts` - Jest testing setup configuration for frontend tests.

### Notes

- Unit tests should be placed alongside the code files they are testing.
- Use `pytest` for backend tests and `jest` for frontend tests.
- Ensure Docker setup for containerized deployment.

## Tasks

- [x] 1.0 Implement Backend Simulator
  - [x] 1.1 Develop core Python module for job generation, queue management, and worker emulation.
    - [x] 1.1.1 Create a `Job` class to represent individual jobs with attributes like size, type, and duration.
    - [x] 1.1.2 Implement a `QueueManager` class to handle job queues and worker assignments.
    - [x] 1.1.3 Develop a `Worker` class to emulate task processing with statistical parameters.
    - [x] 1.1.4 Integrate the `WorkFactory` to generate jobs based on statistical distributions.
  - [x] 1.2 Add support for statistical distributions (Normal, Lognormal, Gamma, Poisson, etc.).
    - [x] 1.2.1 Research and integrate Python libraries for statistical distributions (e.g., `scipy.stats`).
    - [x] 1.2.2 Implement helper functions to generate random values based on the specified distributions.
    - [x] 1.2.3 Write unit tests to validate the correctness of distribution implementations.
  - [x] 1.3 Implement JSON/CSV logging for simulation outputs.
    - [x] 1.3.1 Define the structure of JSON and CSV log files.
    - [x] 1.3.2 Implement logging functionality to record simulation state at each time step.
    - [x] 1.3.3 Ensure logs include all required fields (e.g., timestamps, queue lengths, worker states).
  - [x] 1.4 Create API endpoint for loading scenario files and starting simulations using RestX.
    - [x] 1.4.1 Set up Flask-RESTX for API development.
    - [x] 1.4.2 Implement an endpoint to upload and validate scenario JSON files.
    - [x] 1.4.3 Develop an endpoint to start, pause, and resume simulations.

- [x] 2.0 Develop Frontend Visualization
  - [x] 2.1 Build React-based dashboard for real-time simulation visualization.
    - [x] 2.1.1 Set up the React project structure and install necessary dependencies.
    - [x] 2.1.2 Design the layout for the dashboard, including charts and progress bars.
    - [x] 2.1.3 Implement components for displaying queue lengths, worker utilization, and job progress.
  - [x] 2.2 Integrate WebSocket communication for live updates.
    - [x] 2.2.1 Set up WebSocket communication between the backend and frontend.
    - [x] 2.2.2 Implement real-time updates for dashboard components.
    - [x] 2.2.3 Handle WebSocket connection errors and reconnections gracefully.
  - [x] 2.3 Add controls for pausing, resuming, and stepping through simulation time in specific time periods (a day, a week ...).
    - [x] 2.3.1 Implement buttons for pausing, resuming, and stepping through time.
    - [x] 2.3.2 Ensure backend synchronization with frontend controls.
    - [x] 2.3.3 Test controls for responsiveness and accuracy.

- [x] 3.0 Create Scenario Configuration System
  - [x] 3.1 Define better JSON schema for scenario files. (From /.claude/rules/*.json into his own folder)
    - [x] 3.1.1 Analyze existing JSON files in `/.claude/rules/` for schema design.
    - [x] 3.1.2 Define required fields and their data types in the schema and add them to his own folder.
    - [x] 3.1.3 Validate the schema using sample scenario files.
  - [x] 3.2 Provide predefined scenario templates (MILP, heuristics, ML, mixed).
    - [x] 3.2.1 Create JSON templates for each scenario type.
    - [x] 3.2.2 Document the purpose and usage of each template.
    - [x] 3.2.3 Store templates in the `config/scenarios/` directory.
  - [x] 3.3 Implement validation for scenario files.
    - [x] 3.3.1 Develop a Python script to validate scenario files against the schema.
    - [x] 3.3.2 Integrate validation into the API endpoint for uploading scenarios.
    - [x] 3.3.3 Write tests to ensure validation catches common errors.

- [ ] 4.0 Implement Snapshot and Resume Functionality
  - [ ] 4.1 Add capability to save simulation state to a file.
    - [ ] 4.1.1 Define the structure for snapshot files.
    - [ ] 4.1.2 Implement functionality to serialize and save the simulation state.
    - [ ] 4.1.3 Test saving snapshots under various simulation conditions.
  - [ ] 4.2 Implement functionality to resume simulations from snapshots.
    - [ ] 4.2.1 Develop functionality to load and deserialize snapshot files.
    - [ ] 4.2.2 Ensure the simulator resumes accurately from the loaded state.
    - [ ] 4.2.3 Write tests to validate the correctness of resumed simulations.

- [ ] 5.0 Testing and Deployment
  - [ ] 5.1 Write unit tests for backend and frontend components.
    - [ ] 5.1.1 Identify critical functions and components for testing.
    - [ ] 5.1.2 Write unit tests for backend modules using `pytest`.
    - [ ] 5.1.3 Write unit tests for frontend components using `jest`.
  - [ ] 5.2 Set up Docker for containerized deployment.
    - [ ] 5.2.1 Create a Dockerfile for the backend.
    - [ ] 5.2.2 Create a Dockerfile for the frontend.
    - [ ] 5.2.3 Write a `docker-compose.yml` file to orchestrate services.
  - [ ] 5.3 Validate simulator outputs against predefined success metrics.
    - [ ] 5.3.1 Compare simulation logs against expected outputs.
    - [ ] 5.3.2 Ensure generated datasets meet accuracy requirements for ARIMA/LSTM training.
    - [ ] 5.3.3 Document validation results and address discrepancies.