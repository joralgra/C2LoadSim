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

# Import Redis client
try:
    from redis_client import redis_client
    REDIS_AVAILABLE = True
    print("✅ Redis client imported successfully")
except ImportError as e:
    print(f"⚠️ Redis not available: {e}")
    REDIS_AVAILABLE = False
    redis_client = None

from simulator import (
    SimulationEngine, DistributionType, JobType, JobGenerator,
    Job, Worker, QueueManager, WorkFactory, SimulationSnapshot, SnapshotManager
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
snapshot_ns = Namespace('snapshots', description='Snapshot management operations')

api.add_namespace(simulation_ns, path='/api/simulation')
api.add_namespace(scenario_ns, path='/api/scenarios')
api.add_namespace(data_ns, path='/api/data')
api.add_namespace(health_ns, path='/api')
api.add_namespace(snapshot_ns, path='/api/snapshots')

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
    'duration': fields.Float(description='Simulation duration in seconds (optional with new termination modes)'),
    'num_workers': fields.Integer(description='Number of workers', default=4),
    'queue_config': fields.Raw(description='Queue configuration'),
    'worker_config': fields.Nested(worker_config_model, description='Worker configuration'),
    'job_generation': fields.Raw(description='Job generation parameters'),
    'termination': fields.Raw(description='Termination criteria configuration'),
})

termination_config_model = api.model('TerminationConfig', {
    'mode': fields.String(description='Termination mode', enum=[
        'time', 'data_points', 'jobs', 'completed_jobs'
    ], default='data_points'),
    'duration': fields.Float(description='Duration for time-based termination'),
    'max_data_points': fields.Integer(description='Maximum data points to collect', default=1440),  # 24 hours of minute-by-minute data
    'max_jobs': fields.Integer(description='Maximum jobs to create'),
    'max_completed_jobs': fields.Integer(description='Maximum jobs to complete'),
    'log_interval': fields.Float(description='Logging interval in seconds', default=60.0),  # Log every minute
})

simulation_config_model = api.model('SimulationConfig', {
    'scenario': fields.Nested(scenario_model, required=True),
    'log_directory': fields.String(description='Log output directory', default='logs'),
    'termination': fields.Nested(termination_config_model, description='Termination configuration')
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
            scenario_name = data.get('scenario_name', 'unnamed_scenario')
            termination_config = data.get('termination', {})
            
            # Start Redis session if available
            session_id = ""
            if REDIS_AVAILABLE and redis_client and redis_client.is_connected():
                session_id = redis_client.start_simulation_session(scenario_name, scenario)
                print(f"📝 Started Redis session: {session_id}")
            
            # Create simulation engine with scenario configuration
            simulation_engine = SimulationEngine(
                num_workers=scenario.get('num_workers', 4),
                queue_config=scenario.get('queue_config') or None,
                worker_config=scenario.get('worker_config') or None,
                factory_config=scenario.get('job_generation') or None,
                log_directory=data.get('log_directory', 'logs'),
                termination_config=termination_config
            )
            
            # Start simulation in background thread
            duration = scenario.get('duration')  # May be None for non-time-based termination
            
            def run_simulation():
                try:
                    print(f"Starting simulation '{scenario_name}'")
                    if termination_config.get('mode', 'time') == 'time' and duration:
                        print(f"  Duration: {duration}")
                        result = simulation_engine.run_simulation(duration)
                    else:
                        print(f"  Termination mode: {termination_config.get('mode', 'time')}")
                        result = simulation_engine.run_simulation()
                    print(f"Simulation completed: {result}")
                    
                    # End Redis session when simulation completes
                    if REDIS_AVAILABLE and redis_client and redis_client.is_connected():
                        redis_client.end_simulation_session()
                        print("📝 Ended Redis session")
                        
                except Exception as e:
                    print(f"Simulation error: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # End session even on error
                    if REDIS_AVAILABLE and redis_client and redis_client.is_connected():
                        redis_client.end_simulation_session()
            
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
        
        # Debug information
        print(f"Stop request - simulation_engine exists: {simulation_engine is not None}")
        if simulation_engine:
            print(f"Stop request - simulation_engine.running: {simulation_engine.running}")
        
        if not simulation_engine or not simulation_engine.running:
            api.abort(400, "No simulation is currently running")
        
        simulation_engine.stop_simulation()
        
        # End Redis session if available
        if REDIS_AVAILABLE and redis_client and redis_client.is_connected():
            try:
                redis_client.end_simulation_session()
                print("📝 Ended Redis session on manual stop")
            except Exception as e:
                print(f"Error ending Redis session: {e}")
        
        return {'message': 'Simulation stopped successfully'}


@simulation_ns.route('/redis-status')
class RedisStatus(Resource):
    def get(self):
        """Get Redis connection status and statistics"""
        if not REDIS_AVAILABLE or not redis_client:
            return {
                'available': False,
                'error': 'Redis client not initialized'
            }
        
        stats = redis_client.get_stats()
        return {
            'available': REDIS_AVAILABLE,
            'connected': redis_client.is_connected(),
            'stats': stats
        }


@simulation_ns.route('/clear-redis')
class ClearRedis(Resource):
    def post(self):
        """Clear all simulation data from Redis"""
        if not REDIS_AVAILABLE or not redis_client:
            return {'error': 'Redis not available'}, 400
        
        if redis_client.clear_simulation_data():
            return {'message': 'Redis simulation data cleared successfully'}
        else:
            return {'error': 'Failed to clear Redis data'}, 500
    def get(self):
        """Get historical simulation data from Redis and log files"""
        try:
            history_data = []
            
            # Try Redis first (fastest, most recent data)
            if REDIS_AVAILABLE and redis_client and redis_client.is_connected():
                print("📦 Loading history from Redis...")
                redis_history = redis_client.get_broadcast_history(limit=5000)
                if redis_history:
                    history_data.extend(redis_history)
                    print(f"📦 Loaded {len(redis_history)} entries from Redis")
            
            # Fallback to file-based logs
            if not history_data:
                print("📁 Loading history from log files...")
                log_files = [
                    Path('logs/simulation_log.json'),
                    Path('logs/realtime_simulation_log.jsonl')
                ]
                
                for log_file in log_files:
                    if log_file.exists():
                        print(f"Reading from {log_file}")
                        with open(log_file, 'r') as f:
                            for line in f:
                                try:
                                    data = json.loads(line.strip())
                                    history_data.append(data)
                                except json.JSONDecodeError:
                                    continue  # Skip invalid JSON lines
                
                # Sort by timestamp if we have multiple sources
                if len(history_data) > 1:
                    history_data.sort(key=lambda x: x.get('timestamp', 0))
            
            print(f"📊 Total history entries loaded: {len(history_data)}")
            return {"history": history_data, "source": "redis" if history_data and REDIS_AVAILABLE else "files"}
            
        except Exception as e:
            print(f"❌ Error loading history: {e}")
            return {"error": str(e), "history": []}, 500


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
            from config.validate_scenario import load_schema, validate_scenario_data
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
                        'name': scenario_data.get('scenarioName', 'Unknown'),
                        'description': scenario_data.get('description', ''),
                        'duration': scenario_data.get('duration', 0),
                        'workers': scenario_data.get('workers', {}).get('count', 0),
                        'arrival_type': scenario_data.get('arrival', {}).get('type', 'unknown')
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
            from config.validate_scenario import load_schema, validate_scenario_data
            
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


@scenario_ns.route('/load')
class LoadScenarioByName(Resource):
    @api.expect(api.model('LoadScenario', {
        'scenario': fields.String(required=True, description='Scenario filename')
    }))
    def post(self):
        """Load a scenario by filename and configure the simulation engine."""
        global simulation_engine
        
        data = request.get_json()
        filename = data.get('scenario')
        
        if not filename:
            api.abort(400, "Scenario filename is required")
        
        scenario_file = Path('config/scenarios') / filename
        
        if not scenario_file.exists():
            api.abort(404, "Scenario not found")
        
        # Stop any running simulation first
        if simulation_engine and simulation_engine.is_running():
            return {'error': 'Cannot load scenario while simulation is running. Please stop the simulation first.'}, 400
        
        try:
            with open(scenario_file, 'r') as f:
                scenario_data = json.load(f)
            
            # Convert scenario format to simulation engine format
            worker_config = scenario_data.get('workers', {})
            speed_dist = worker_config.get('speedDistribution', {})
            
            # Create simulation engine with scenario configuration
            simulation_engine = SimulationEngine(
                num_workers=worker_config.get('count', 4),
                queue_config={
                    'max_queue_size': 1000,
                    'use_priority_queue': False
                },
                worker_config={
                    'processing_speed': speed_dist.get('mu', 1.0) if speed_dist.get('type') == 'normal' else 1.0,
                    'failure_rate': 0.02,
                    'efficiency_variance': speed_dist.get('sigma', 0.1) if speed_dist.get('type') == 'normal' else 0.1,
                    'max_concurrent_jobs': 2
                },
                factory_config={
                    'arrival_pattern': scenario_data.get('arrival', {}),
                    'job_size_distribution': scenario_data.get('jobSize', {}),
                    'job_types': {
                        'milp': 0.25,
                        'heuristic': 0.35,
                        'ml': 0.25,
                        'mixed': 0.15
                    }
                }
            )
            
            return {
                'message': f'Scenario "{scenario_data.get("scenarioName", "Unknown")}" loaded successfully',
                'scenario_name': scenario_data.get('scenarioName', 'Unknown'),
                'duration': scenario_data.get('duration', 0),
                'workers': worker_config.get('count', 4),
                'arrival_type': scenario_data.get('arrival', {}).get('type', 'unknown')
            }
            
        except Exception as e:
            api.abort(500, f"Failed to load scenario: {e}")


@scenario_ns.route('/load/<string:filename>')
class LoadScenario(Resource):
    def post(self, filename):
        """Load a scenario and configure the simulation engine."""
        global simulation_engine
        
        scenario_file = Path('config/scenarios') / filename
        
        if not scenario_file.exists():
            api.abort(404, "Scenario not found")
        
        # Stop any running simulation first
        if simulation_engine and simulation_engine.is_running():
            return {'error': 'Cannot load scenario while simulation is running. Please stop the simulation first.'}, 400
        
        try:
            with open(scenario_file, 'r') as f:
                scenario_data = json.load(f)
            
            # Convert scenario format to simulation engine format
            worker_config = scenario_data.get('workers', {})
            speed_dist = worker_config.get('speedDistribution', {})
            
            # Create simulation engine with scenario configuration
            simulation_engine = SimulationEngine(
                num_workers=worker_config.get('count', 4),
                queue_config={
                    'max_queue_size': 1000,
                    'use_priority_queue': False
                },
                worker_config={
                    'processing_speed': speed_dist.get('mu', 1.0) if speed_dist.get('type') == 'normal' else 1.0,
                    'failure_rate': 0.02,
                    'efficiency_variance': speed_dist.get('sigma', 0.1) if speed_dist.get('type') == 'normal' else 0.1,
                    'max_concurrent_jobs': 2
                },
                factory_config={
                    'arrival_pattern': scenario_data.get('arrival', {}),
                    'job_size_distribution': scenario_data.get('jobSize', {}),
                    'job_types': {
                        'milp': 0.25,
                        'heuristic': 0.35,
                        'ml': 0.25,
                        'mixed': 0.15
                    }
                }
            )
            
            return {
                'message': f'Scenario "{scenario_data.get("scenarioName", "Unknown")}" loaded successfully',
                'scenario_name': scenario_data.get('scenarioName', 'Unknown'),
                'duration': scenario_data.get('duration', 0),
                'workers': worker_config.get('count', 4),
                'arrival_type': scenario_data.get('arrival', {}).get('type', 'unknown')
            }
            
        except Exception as e:
            api.abort(500, f"Failed to load scenario: {e}")


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


@data_ns.route('/history')
class SimulationHistory(Resource):
    def get(self):
        """Get historical simulation data from Redis or file logs."""
        global redis_client
        
        # Try Redis first for high-performance access
        if REDIS_AVAILABLE and redis_client and redis_client.is_connected():
            try:
                history_data = redis_client.get_simulation_history()
                if history_data:
                    return {
                        'data': history_data,
                        'source': 'redis',
                        'count': len(history_data)
                    }
            except Exception as e:
                print(f"Error retrieving history from Redis: {e}")
        
        # Fallback to file logs
        try:
            # Read from JSON log file as fallback
            json_log_path = "logs/simulation_log.json"
            if os.path.exists(json_log_path):
                with open(json_log_path, 'r') as f:
                    file_data = json.load(f)
                return {
                    'data': file_data,
                    'source': 'file',
                    'count': len(file_data) if isinstance(file_data, list) else 1
                }
            else:
                return {
                    'data': [],
                    'source': 'none',
                    'count': 0,
                    'message': 'No simulation history available'
                }
        except Exception as e:
            return {
                'data': [],
                'source': 'error',
                'count': 0,
                'error': str(e)
            }


# Health check endpoint
@health_ns.route('/health')
class Health(Resource):
    def get(self):
        """Health check endpoint."""
        return {'status': 'healthy', 'message': 'C2LoadSim API is running'}


# Snapshot management endpoints
@snapshot_ns.route('/create')
class CreateSnapshot(Resource):
    def post(self):
        """Create a snapshot of the current simulation state."""
        global simulation_engine
        
        if not simulation_engine:
            api.abort(404, "No simulation instance available")
        
        data = request.json or {}
        description = data.get('description', 'Manual snapshot')
        filename = data.get('filename', None)
        
        try:
            snapshot_path = simulation_engine.save_snapshot(description, filename)
            snapshot_info = simulation_engine.create_snapshot(description)
            
            return {
                'success': True,
                'message': 'Snapshot created successfully',
                'snapshot_path': snapshot_path,
                'snapshot_id': snapshot_info.snapshot_id,
                'timestamp': snapshot_info.timestamp.isoformat(),
                'simulation_time': snapshot_info.simulation_time,
                'description': description
            }
        except Exception as e:
            api.abort(500, f"Failed to create snapshot: {e}")


@snapshot_ns.route('/list')
class ListSnapshots(Resource):
    def get(self):
        """List available snapshots."""
        global simulation_engine
        
        if not simulation_engine:
            api.abort(404, "No simulation instance available")
        
        try:
            snapshots = simulation_engine.list_snapshots()
            return {
                'snapshots': snapshots,
                'total_count': len(snapshots)
            }
        except Exception as e:
            api.abort(500, f"Failed to list snapshots: {e}")


@snapshot_ns.route('/restore/<string:filename>')
class RestoreSnapshot(Resource):
    def post(self, filename):
        """Restore simulation state from a snapshot."""
        global simulation_engine
        
        if not simulation_engine:
            api.abort(404, "No simulation instance available")
        
        if simulation_engine.running:
            api.abort(400, "Cannot restore while simulation is running. Stop it first.")
        
        try:
            result = simulation_engine.restore_from_snapshot(filename)
            
            if result.get("success"):
                return {
                    'message': 'Simulation restored from snapshot successfully',
                    'result': result
                }
            else:
                api.abort(400, result.get("error", "Failed to restore from snapshot"))
                
        except FileNotFoundError:
            api.abort(404, f"Snapshot file not found: {filename}")
        except Exception as e:
            api.abort(500, f"Failed to restore from snapshot: {e}")


@snapshot_ns.route('/delete/<string:filename>')
class DeleteSnapshot(Resource):
    def delete(self, filename):
        """Delete a snapshot file."""
        global simulation_engine
        
        if not simulation_engine:
            api.abort(404, "No simulation instance available")
        
        try:
            success = simulation_engine.delete_snapshot(filename)
            
            if success:
                return {'message': f'Snapshot {filename} deleted successfully'}
            else:
                api.abort(404, f"Snapshot file not found: {filename}")
                
        except Exception as e:
            api.abort(500, f"Failed to delete snapshot: {e}")


@snapshot_ns.route('/download/<string:filename>')
class DownloadSnapshot(Resource):
    def get(self, filename):
        """Download a snapshot file."""
        global simulation_engine
        
        if not simulation_engine:
            api.abort(404, "No simulation instance available")
        
        try:
            snapshot_path = Path('snapshots') / filename
            
            if not snapshot_path.exists():
                api.abort(404, f"Snapshot file not found: {filename}")
            
            return send_file(snapshot_path, as_attachment=True, download_name=filename)
        except Exception as e:
            api.abort(500, f"Failed to download snapshot: {e}")


@data_ns.route('/logs/detailed')
class DetailedLogs(Resource):
    def get(self):
        """Download detailed JSON log file."""
        global simulation_engine
        
        if not simulation_engine:
            api.abort(404, "No simulation data available")
        
        detailed_file = simulation_engine.logger.detailed_log_file
        
        if not detailed_file.exists():
            api.abort(404, "Detailed log file not found")
        
        return send_file(detailed_file, as_attachment=True, download_name='detailed_simulation_log.json')


@data_ns.route('/logs/metrics')
class MetricsLogs(Resource):
    def get(self):
        """Download metrics CSV log file."""
        global simulation_engine
        
        if not simulation_engine:
            api.abort(404, "No simulation data available")
        
        metrics_file = simulation_engine.logger.metrics_log_file
        
        if not metrics_file.exists():
            api.abort(404, "Metrics log file not found")
        
        return send_file(metrics_file, as_attachment=True, download_name='metrics_log.csv')


@data_ns.route('/logs/statistics')
class LogStatistics(Resource):
    def get(self):
        """Get comprehensive log statistics."""
        global simulation_engine
        
        if not simulation_engine:
            api.abort(404, "No simulation data available")
        
        return simulation_engine.logger.get_log_statistics()


@data_ns.route('/logs/backup')
class BackupLogs(Resource):
    def post(self):
        """Create backup copies of all log files."""
        global simulation_engine
        
        if not simulation_engine:
            api.abort(404, "No simulation data available")
        
        data = request.json or {}
        backup_suffix = data.get('backup_suffix', None)
        
        try:
            backup_paths = simulation_engine.logger.backup_logs(backup_suffix)
            
            return {
                'message': 'Log backup created successfully',
                'backup_files': backup_paths,
                'backup_count': len(backup_paths)
            }
        except Exception as e:
            api.abort(500, f"Failed to create log backup: {e}")


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
            
            # Create update payload
            update_payload = {
                'status': status,
                'queue_stats': queue_stats,
                'worker_stats': worker_stats,
                'timestamp': time.time()
            }
            
            # Broadcast the update
            socketio.emit('simulation_update', update_payload)
            
            # Store to persistence layer
            store_broadcast_data(update_payload)
        
        # Wait before next update
        socketio.sleep(1)  # Update every second


def store_broadcast_data(update_payload: Dict[str, Any]) -> None:
    """
    Store broadcast data to available persistence layers.
    
    Uses Redis for high performance, falls back to existing logger system.
    """
    stored = False
    
    # Try Redis first - pass the simulation engine directly
    if REDIS_AVAILABLE and redis_client and redis_client.is_connected() and simulation_engine:
        try:
            # Store using the simulation engine (as expected by the Redis method)
            stored = redis_client.store_simulation_state(simulation_engine)
            if stored:
                print("📦 Stored comprehensive simulation state to Redis")
        except Exception as e:
            print(f"Error storing to Redis: {e}")
    
    # Fallback to existing logger system
    if not stored and simulation_engine and hasattr(simulation_engine, 'logger'):
        try:
            # Create a simplified log entry for the existing logger
            broadcast_log_entry = {
                'timestamp': update_payload['timestamp'],
                'simulation_time': update_payload['status']['simulation_time'],
                'total_jobs_created': update_payload['status']['total_jobs_created'],
                'queue_length': update_payload['status']['queue_length'],
                'active_workers': update_payload['status']['active_workers'],
                'running': update_payload['status']['running'],
                'paused': update_payload['status']['paused'],
            }
            
            # Use the existing JSON logging method instead of non-existent log_broadcast_data
            if hasattr(simulation_engine.logger, 'append_to_json_log_file'):
                simulation_engine.logger.append_to_json_log_file(broadcast_log_entry)
                print("📝 Stored to file logger (fallback)")
        except Exception as e:
            print(f"Error storing to file logger: {e}")
            print(f"❌ Failed to store broadcast data: {e}")


if __name__ == '__main__':
    # Create necessary directories
    Path('config/scenarios').mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    Path('snapshots').mkdir(exist_ok=True)
    
    # Start the background task for broadcasting updates
    socketio.start_background_task(broadcast_updates)
    
    # Run the Flask app with SocketIO
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
