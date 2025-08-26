"""
C2LoadSim REST API

Flask-RESTX based API for controlling simulations, uploading scenarios,
and retrieving simulation data.
"""

from flask import Flask, request, jsonify, send_file
from flask_restx import Api, Resource, fields, Namespace
from flask_socketio import SocketIO, emit
from werkzeug.exceptions import BadRequest, NotFound
import json
import threading
import time
from pathlib import Path
import tempfile
from typing import Dict, Any, Optional

import sys
import os
sys.path.append(os.path.dirname(__file__))

from simulator import (
    SimulationEngine, DistributionType, JobType, JobGenerator,
    Job, Worker, QueueManager, WorkFactory
)


# Initialize Flask app
app = Flask(__name__)
app.config['RESTX_MASK_SWAGGER'] = False
app.config['SECRET_KEY'] = 'c2loadsim-secret-key'

# Handle CORS for all requests
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response

# Handle preflight OPTIONS requests
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = app.response_class()
        response.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        return response

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize Flask-RESTX API
api = Api(
    app,
    version='1.0',
    title='C2LoadSim API',
    description='Computation Load Simulation API for generating and managing workload simulations',
    doc='/docs/'
)

# Global simulation engine instance
simulation_engine: Optional[SimulationEngine] = None
simulation_thread: Optional[threading.Thread] = None

# API Namespaces
simulation_ns = Namespace('simulation', description='Simulation control operations')
scenario_ns = Namespace('scenarios', description='Scenario management operations')
data_ns = Namespace('data', description='Data retrieval operations')
health_ns = Namespace('health', description='Health check operations')

api.add_namespace(simulation_ns, path='/api/simulation')
api.add_namespace(scenario_ns, path='/api/scenarios')
api.add_namespace(data_ns, path='/api/data')
api.add_namespace(health_ns, path='/api')

# API Models for documentation
job_model = api.model('Job', {
    'job_id': fields.String(description='Unique job identifier'),
    'job_type': fields.String(description='Type of job', enum=['milp', 'heuristic', 'ml', 'mixed']),
    'size': fields.Float(description='Job size/complexity'),
    'duration': fields.Float(description='Expected processing duration'),
    'priority': fields.Integer(description='Job priority'),
    'status': fields.String(description='Job status', enum=['pending', 'running', 'completed', 'failed']),
})

worker_config_model = api.model('WorkerConfig', {
    'processing_speed': fields.Float(description='Worker processing speed multiplier', default=1.0),
    'failure_rate': fields.Float(description='Job failure probability', default=0.01),
    'efficiency_variance': fields.Float(description='Processing efficiency variance', default=0.1),
    'max_concurrent_jobs': fields.Integer(description='Maximum concurrent jobs', default=1),
})

distribution_model = api.model('Distribution', {
    'distribution': fields.String(description='Distribution type', enum=[
        'normal', 'lognormal', 'exponential', 'uniform', 'gamma', 'poisson', 'constant'
    ]),
    'params': fields.Raw(description='Distribution parameters')
})

scenario_model = api.model('Scenario', {
    'name': fields.String(required=True, description='Scenario name'),
    'description': fields.String(description='Scenario description'),
    'duration': fields.Float(required=True, description='Simulation duration in seconds'),
    'num_workers': fields.Integer(description='Number of workers', default=4),
    'queue_config': fields.Raw(description='Queue configuration'),
    'worker_config': fields.Nested(worker_config_model, description='Worker configuration'),
    'job_generation': fields.Raw(description='Job generation parameters'),
})

simulation_config_model = api.model('SimulationConfig', {
    'scenario': fields.Nested(scenario_model, required=True),
    'log_directory': fields.String(description='Log output directory', default='logs')
})

simulation_status_model = api.model('SimulationStatus', {
    'running': fields.Boolean(description='Whether simulation is running'),
    'paused': fields.Boolean(description='Whether simulation is paused'),
    'simulation_time': fields.Float(description='Current simulation time'),
    'total_jobs_created': fields.Integer(description='Total jobs created'),
    'queue_length': fields.Integer(description='Current queue length'),
    'active_workers': fields.Integer(description='Number of active workers'),
})


@simulation_ns.route('/start')
class StartSimulation(Resource):
    @api.expect(simulation_config_model)
    @api.marshal_with(simulation_status_model)
    def post(self):
        """Start a new simulation with the provided configuration."""
        global simulation_engine, simulation_thread
        
        if simulation_engine and simulation_engine.running:
            api.abort(400, "Simulation is already running")
        
        try:
            data = request.json
            scenario = data['scenario']
            
            # Create simulation engine with scenario configuration
            simulation_engine = SimulationEngine(
                num_workers=scenario.get('num_workers', 4),
                queue_config=scenario.get('queue_config') or None,
                worker_config=scenario.get('worker_config') or None,
                factory_config=scenario.get('job_generation') or None,
                log_directory=data.get('log_directory', 'logs')
            )
            
            # Start simulation in background thread
            duration = scenario['duration']
            
            def run_simulation():
                try:
                    print(f"Starting simulation with duration: {duration}")
                    result = simulation_engine.run_simulation(duration)
                    print(f"Simulation completed: {result}")
                except Exception as e:
                    print(f"Simulation error: {e}")
                    import traceback
                    traceback.print_exc()
            
            simulation_thread = threading.Thread(target=run_simulation)
            simulation_thread.daemon = True
            simulation_thread.start()
            
            return {
                'running': True,
                'paused': False,
                'simulation_time': 0.0,
                'total_jobs_created': 0,
                'queue_length': 0,
                'active_workers': 0
            }
            
        except KeyError as e:
            api.abort(400, f"Missing required field: {e}")
        except Exception as e:
            api.abort(500, f"Failed to start simulation: {e}")


@simulation_ns.route('/status')
class SimulationStatus(Resource):
    @api.marshal_with(simulation_status_model)
    def get(self):
        """Get current simulation status."""
        global simulation_engine
        
        if not simulation_engine:
            return {
                'running': False,
                'paused': False,
                'simulation_time': 0.0,
                'total_jobs_created': 0,
                'queue_length': 0,
                'active_workers': 0
            }
        
        return {
            'running': simulation_engine.running,
            'simulation_time': simulation_engine.simulation_time,
            'total_jobs_created': simulation_engine.total_jobs_created,
            'queue_length': simulation_engine.queue_manager.get_queue_length(),
            'active_workers': len([w for w in simulation_engine.workers if w.status.value == 'busy']),
            'paused': getattr(simulation_engine, 'paused', False)
        }


@simulation_ns.route('/stop')
class StopSimulation(Resource):
    def post(self):
        """Stop the running simulation."""
        global simulation_engine
        
        if not simulation_engine or not simulation_engine.running:
            api.abort(400, "No simulation is currently running")
        
        simulation_engine.stop_simulation()
        
        return {'message': 'Simulation stopped successfully'}


@simulation_ns.route('/reset')
class ResetSimulation(Resource):
    def post(self):
        """Reset simulation to initial state."""
        global simulation_engine
        
        if not simulation_engine:
            api.abort(400, "No simulation instance available")
        
        if simulation_engine.running:
            api.abort(400, "Cannot reset while simulation is running. Stop it first.")
        
        simulation_engine.reset_simulation()
        
        return {'message': 'Simulation reset successfully'}


@simulation_ns.route('/pause')
class PauseSimulation(Resource):
    def post(self):
        """Pause the running simulation."""
        global simulation_engine
        
        if not simulation_engine or not simulation_engine.running:
            api.abort(400, "No simulation is currently running")
        
        if getattr(simulation_engine, 'paused', False):
            api.abort(400, "Simulation is already paused")
        
        simulation_engine.pause_simulation()
        
        return {'message': 'Simulation paused successfully'}


@simulation_ns.route('/resume')
class ResumeSimulation(Resource):
    def post(self):
        """Resume the paused simulation."""
        global simulation_engine
        
        if not simulation_engine or not simulation_engine.running:
            api.abort(400, "No simulation is currently running")
        
        if not getattr(simulation_engine, 'paused', False):
            api.abort(400, "Simulation is not paused")
        
        simulation_engine.resume_simulation()
        
        return {'message': 'Simulation resumed successfully'}


@simulation_ns.route('/step')
class StepSimulation(Resource):
    def post(self):
        """Execute a single simulation step while paused."""
        global simulation_engine
        
        if not simulation_engine:
            api.abort(400, "No simulation instance available")
        
        result = simulation_engine.step_once()
        
        if 'error' in result:
            api.abort(400, result['error'])
        
        return result


step_time_model = api.model('StepTimeRequest', {
    'time_period': fields.Float(required=True, description='Time period to step forward (seconds)'),
    'unit': fields.String(description='Time unit (seconds, minutes, hours, days, weeks)', default='seconds')
})


@simulation_ns.route('/step-time')
class StepSimulationTime(Resource):
    @api.expect(step_time_model)
    def post(self):
        """Execute simulation steps for a specific time period while paused."""
        global simulation_engine
        
        if not simulation_engine:
            api.abort(400, "No simulation instance available")
        
        data = request.json
        time_period = data.get('time_period', 0)
        unit = data.get('unit', 'seconds')
        
        # Convert time period to seconds
        unit_multipliers = {
            'seconds': 1,
            'minutes': 60,
            'hours': 3600,
            'days': 86400,
            'weeks': 604800
        }
        
        if unit not in unit_multipliers:
            api.abort(400, f"Invalid time unit. Must be one of: {list(unit_multipliers.keys())}")
        
        time_period_seconds = time_period * unit_multipliers[unit]
        
        result = simulation_engine.step_time_period(time_period_seconds)
        
        if 'error' in result:
            api.abort(400, result['error'])
        
        return {
            **result,
            'requested_time_period': time_period,
            'requested_unit': unit,
            'time_period_seconds': time_period_seconds
        }


@scenario_ns.route('/upload')
class UploadScenario(Resource):
    def post(self):
        """Upload and validate a scenario file."""
        if 'file' not in request.files:
            api.abort(400, "No file provided")
        
        file = request.files['file']
        if file.filename == '':
            api.abort(400, "No file selected")
        
        if not file.filename.endswith('.json'):
            api.abort(400, "File must be a JSON file")
        
        try:
            # Read and parse JSON
            content = file.read().decode('utf-8')
            scenario_data = json.loads(content)
            
            # Validate against schema
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
            from validate_scenario import load_schema, validate_scenario_data
            schema = load_schema()
            is_valid, validation_errors = validate_scenario_data(scenario_data, schema)
            
            if not is_valid:
                return {
                    'error': 'Scenario validation failed',
                    'validation_errors': validation_errors
                }, 400
            
            # Save scenario file
            scenarios_dir = Path('config/scenarios')
            scenarios_dir.mkdir(parents=True, exist_ok=True)
            
            scenario_file = scenarios_dir / file.filename
            with open(scenario_file, 'w') as f:
                json.dump(scenario_data, f, indent=2)
            
            return {
                'message': 'Scenario uploaded and validated successfully',
                'filename': file.filename,
                'scenario_name': scenario_data['scenarioName']
            }
            
        except json.JSONDecodeError:
            api.abort(400, "Invalid JSON format")
        except Exception as e:
            api.abort(500, f"Failed to upload scenario: {e}")


@scenario_ns.route('/list')
class ListScenarios(Resource):
    def get(self):
        """List available scenario files."""
        scenarios_dir = Path('config/scenarios')
        
        if not scenarios_dir.exists():
            return {'scenarios': []}
        
        scenarios = []
        for scenario_file in scenarios_dir.glob('*.json'):
            try:
                with open(scenario_file, 'r') as f:
                    scenario_data = json.load(f)
                    scenarios.append({
                        'filename': scenario_file.name,
                        'name': scenario_data.get('name', 'Unknown'),
                        'description': scenario_data.get('description', ''),
                        'duration': scenario_data.get('duration', 0)
                    })
            except Exception:
                # Skip invalid files
                continue
        
        return {'scenarios': scenarios}


@scenario_ns.route('/validate')
class ValidateScenario(Resource):
    def post(self):
        """Validate a scenario JSON without uploading it."""
        data = request.json
        
        if not data:
            api.abort(400, "No JSON data provided")
        
        try:
            # Import validation functions
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
            from validate_scenario import load_schema, validate_scenario_data
            
            # Load schema and validate
            schema = load_schema()
            is_valid, validation_errors = validate_scenario_data(data, schema)
            
            if is_valid:
                return {
                    'valid': True,
                    'message': 'Scenario is valid',
                    'scenario_name': data.get('scenarioName', 'Unknown')
                }
            else:
                return {
                    'valid': False,
                    'message': 'Scenario validation failed',
                    'validation_errors': validation_errors
                }, 400
                
        except Exception as e:
            api.abort(500, f"Failed to validate scenario: {e}")


@scenario_ns.route('/<string:filename>')
class GetScenario(Resource):
    def get(self, filename):
        """Get a specific scenario by filename."""
        scenario_file = Path('config/scenarios') / filename
        
        if not scenario_file.exists():
            api.abort(404, "Scenario not found")
        
        try:
            with open(scenario_file, 'r') as f:
                scenario_data = json.load(f)
            return scenario_data
        except Exception as e:
            api.abort(500, f"Failed to read scenario: {e}")


@data_ns.route('/logs/summary')
class LogSummary(Resource):
    def get(self):
        """Get simulation log summary."""
        global simulation_engine
        
        if not simulation_engine:
            api.abort(404, "No simulation data available")
        
        return simulation_engine.logger.get_log_summary()


@data_ns.route('/logs/json')
class JSONLogs(Resource):
    def get(self):
        """Download JSON log file."""
        global simulation_engine
        
        if not simulation_engine:
            api.abort(404, "No simulation data available")
        
        json_file = simulation_engine.logger.json_log_file
        
        if not json_file.exists():
            api.abort(404, "JSON log file not found")
        
        return send_file(json_file, as_attachment=True, download_name='simulation_log.json')


@data_ns.route('/logs/csv')
class CSVLogs(Resource):
    def get(self):
        """Download CSV log file."""
        global simulation_engine
        
        if not simulation_engine:
            api.abort(404, "No simulation data available")
        
        csv_file = simulation_engine.logger.csv_log_file
        
        if not csv_file.exists():
            api.abort(404, "CSV log file not found")
        
        return send_file(csv_file, as_attachment=True, download_name='simulation_log.csv')


@data_ns.route('/queue/stats')
class QueueStats(Resource):
    def get(self):
        """Get current queue statistics."""
        global simulation_engine
        
        if not simulation_engine:
            api.abort(404, "No simulation data available")
        
        return simulation_engine.queue_manager.get_queue_stats()


@data_ns.route('/workers/stats')
class WorkerStats(Resource):
    def get(self):
        """Get current worker statistics."""
        global simulation_engine
        
        if not simulation_engine:
            api.abort(404, "No simulation data available")
        
        return {
            'workers': [worker.get_stats() for worker in simulation_engine.workers]
        }


# Health check endpoint
@health_ns.route('/health')
class Health(Resource):
    def get(self):
        """Health check endpoint."""
        return {'status': 'healthy', 'message': 'C2LoadSim API is running'}


# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print(f'Client connected')
    emit('connected', {'message': 'Connected to C2LoadSim'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('Client disconnected')


@socketio.on('join_simulation')
def handle_join_simulation():
    """Join simulation updates room."""
    print('Client joined simulation updates')
    emit('joined_simulation', {'message': 'Joined simulation updates'})


# Background task for broadcasting updates
def broadcast_updates():
    """Background task to broadcast simulation updates to all connected clients."""
    while True:
        if simulation_engine and simulation_engine.running:
            # Get current status
            status = {
                'running': simulation_engine.running,
                'paused': getattr(simulation_engine, 'paused', False),
                'simulation_time': simulation_engine.simulation_time,
                'total_jobs_created': simulation_engine.total_jobs_created,
                'queue_length': simulation_engine.queue_manager.get_queue_length(),
                'active_workers': len([w for w in simulation_engine.workers if w.status.value == 'busy'])
            }
            
            # Get queue stats
            queue_stats = simulation_engine.queue_manager.get_queue_stats()
            
            # Get worker stats
            worker_stats = [worker.get_stats() for worker in simulation_engine.workers]
            
            # Broadcast the update
            socketio.emit('simulation_update', {
                'status': status,
                'queue_stats': queue_stats,
                'worker_stats': worker_stats,
                'timestamp': time.time()
            })
        
        # Wait before next update
        socketio.sleep(1)  # Update every second


if __name__ == '__main__':
    # Create necessary directories
    Path('config/scenarios').mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    
    # Start the background task for broadcasting updates
    socketio.start_background_task(broadcast_updates)
    
    # Run the Flask app with SocketIO
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
