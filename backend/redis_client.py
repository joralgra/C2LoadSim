"""
Redis client and utilities for C2LoadSim

Provides Redis connection management and data storage utilities
for real-time simulation data persistence.
"""

import redis
import json
import time
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

# Custom JSON encoder to handle Enum serialization
class SimulationJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for simulation objects."""
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class RedisClient:
    """Redis client for C2LoadSim simulation data storage."""
    
    def __init__(self, 
                 host: str = 'localhost', 
                 port: int = 6379, 
                 db: int = 0,
                 decode_responses: bool = True):
        """
        Initialize Redis client.
        
        Args:
            host: Redis server hostname
            port: Redis server port
            db: Redis database number
            decode_responses: Whether to decode responses as strings
        """
        self.host = host
        self.port = port
        self.db = db
        
        # Environment variable overrides
        self.host = os.getenv('REDIS_HOST', self.host)
        self.port = int(os.getenv('REDIS_PORT', self.port))
        
        try:
            self.redis = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=decode_responses,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            self.redis.ping()
            print(f"✅ Redis connected: {self.host}:{self.port}")
            
        except redis.ConnectionError as e:
            print(f"❌ Redis connection failed: {e}")
            self.redis = None
    
    def is_connected(self) -> bool:
        """Check if Redis is connected and available."""
        if not self.redis:
            return False
        try:
            self.redis.ping()
            return True
        except:
            return False
    
    def store_simulation_state(self, simulation_engine) -> bool:
        """
        Store complete simulation state to Redis.
        
        Args:
            simulation_engine: The simulation engine instance
            
        Returns:
            bool: Success status
        """
        if not self.is_connected():
            return False
            
        try:
            timestamp = int(time.time() * 1000)
            
            # Create comprehensive simulation state
            simulation_state = {
                'timestamp': timestamp / 1000,  # Convert back to seconds for consistency
                'simulation_time': simulation_engine.simulation_time,
                'running': simulation_engine.running,
                'paused': getattr(simulation_engine, 'paused', False),
                'total_jobs_created': simulation_engine.total_jobs_created,
                'queue_length': simulation_engine.queue_manager.get_queue_length(),
                'active_workers': len([w for w in simulation_engine.workers if w.status.value == 'busy']),
                
                # Detailed queue statistics
                'queue_stats': simulation_engine.queue_manager.get_queue_stats(),
                
                # Individual worker statistics
                'worker_stats': {
                    'individual_utilizations': [],
                    'worker_details': []
                },
                
                # Performance metrics
                'performance_metrics': {
                    'throughput': 0,
                    'avg_response_time': 0,
                    'success_rate': 100
                }
            }
            
            # Collect individual worker data
            for i, worker in enumerate(simulation_engine.workers):
                worker_stats = worker.get_stats()
                simulation_state['worker_stats']['individual_utilizations'].append(
                    worker_stats.get('utilization', 0) / 100.0  # Convert percentage to decimal
                )
                simulation_state['worker_stats']['worker_details'].append({
                    'worker_id': f'worker_{i}',
                    'worker_index': i + 1,
                    'status': worker.status.value,
                    'utilization': worker_stats.get('utilization', 0),
                    'completed_jobs': worker_stats.get('completed_jobs', 0),
                    'failed_jobs': worker_stats.get('failed_jobs', 0),
                    'current_jobs': worker_stats.get('current_jobs', 0),
                    'average_processing_time': worker_stats.get('average_processing_time', 0)
                })
            
            # Calculate aggregated metrics
            if simulation_state['worker_stats']['individual_utilizations']:
                avg_util = sum(simulation_state['worker_stats']['individual_utilizations']) / len(simulation_state['worker_stats']['individual_utilizations'])
                simulation_state['performance_metrics']['average_utilization'] = avg_util * 100  # Convert back to percentage
            
            # Store in multiple Redis structures for different access patterns
            
            # 1. Time-series data (sorted set)
            self.redis.zadd('simulation:timeseries', {
                json.dumps(simulation_state, cls=SimulationJSONEncoder): timestamp
            })
            
            # 2. Latest state (string with TTL)
            self.redis.setex('simulation:current_state', 3600, json.dumps(simulation_state, cls=SimulationJSONEncoder))
            
            # 3. Individual metrics for analysis
            metrics_to_store = {
                'jobs_created': simulation_state['total_jobs_created'],
                'queue_length': simulation_state['queue_length'],
                'active_workers': simulation_state['active_workers'],
                'avg_utilization': simulation_state['performance_metrics'].get('average_utilization', 0)
            }
            
            for metric_name, value in metrics_to_store.items():
                if isinstance(value, (int, float)):
                    self.redis.zadd(f'metrics:{metric_name}', {str(value): timestamp})
            
            # Clean up old data (keep last 4 hours of detailed data)
            four_hours_ago = timestamp - (4 * 3600 * 1000)
            self.redis.zremrangebyscore('simulation:timeseries', 0, four_hours_ago)
            
            # Clean up individual metrics (keep last 24 hours)
            twenty_four_hours_ago = timestamp - (24 * 3600 * 1000)
            for metric_name in metrics_to_store.keys():
                self.redis.zremrangebyscore(f'metrics:{metric_name}', 0, twenty_four_hours_ago)
            
            return True
            
        except Exception as e:
            print(f"Error storing simulation state to Redis: {e}")
            return False
        """
        Store broadcast update as time-series data.
        
        Args:
            update_payload: The broadcast data to store
            
        Returns:
            bool: Success status
        """
        if not self.is_connected():
            return False
        
        try:
            timestamp = int(time.time() * 1000)  # milliseconds
            
            # Store in sorted set with timestamp as score
            self.redis.zadd('simulation:broadcast', {
                json.dumps(update_payload): timestamp
            })
            
            # Also store latest state for quick access
            self.redis.set('simulation:latest', json.dumps(update_payload))
            self.redis.expire('simulation:latest', 3600)  # Expire after 1 hour
            
            # Keep only last 2 hours of broadcast data
            two_hours_ago = timestamp - (2 * 3600 * 1000)
            self.redis.zremrangebyscore('simulation:broadcast', 0, two_hours_ago)
            
            return True
            
        except Exception as e:
            print(f"Error storing to Redis: {e}")
            return False
    
    def get_simulation_history(self, 
                                limit: int = 2000, 
                                start_time: Optional[float] = None,
                                end_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Retrieve simulation history from Redis in the format expected by the frontend.
        
        Args:
            limit: Maximum number of records to return
            start_time: Start timestamp (Unix timestamp)
            end_time: End timestamp (Unix timestamp)
            
        Returns:
            List of simulation history records formatted for frontend
        """
        if not self.is_connected():
            return []
        
        try:
            # Convert to milliseconds if provided
            if start_time:
                start_time = int(start_time * 1000)
            else:
                start_time = 0
                
            if end_time:
                end_time = int(end_time * 1000)
            else:
                end_time = int(time.time() * 1000)
            
            # Get data from time-series sorted set
            raw_data = self.redis.zrangebyscore(
                'simulation:timeseries',
                start_time,
                end_time,
                start=0,
                num=limit
            )
            
            # Parse and format data for frontend
            history = []
            for item in raw_data:
                try:
                    data = json.loads(item)
                    history.append(data)
                except json.JSONDecodeError:
                    continue
            
            print(f"📦 Retrieved {len(history)} simulation records from Redis")
            return history
            
        except Exception as e:
            print(f"Error retrieving simulation history from Redis: {e}")
            return []
        """
        Retrieve broadcast history from Redis.
        
        Args:
            limit: Maximum number of records to return
            start_time: Start timestamp (Unix timestamp)
            end_time: End timestamp (Unix timestamp)
            
        Returns:
            List of broadcast data records
        """
        if not self.is_connected():
            return []
        
        try:
            # Convert to milliseconds if provided
            if start_time:
                start_time = int(start_time * 1000)
            else:
                start_time = 0
                
            if end_time:
                end_time = int(end_time * 1000)
            else:
                end_time = int(time.time() * 1000)
            
            # Get data from sorted set
            raw_data = self.redis.zrangebyscore(
                'simulation:broadcast',
                start_time,
                end_time,
                start=0,
                num=limit
            )
            
            # Parse JSON data
            history = []
            for item in raw_data:
                try:
                    data = json.loads(item)
                    history.append(data)
                except json.JSONDecodeError:
                    continue
            
            return history
            
        except Exception as e:
            print(f"Error retrieving from Redis: {e}")
            return []
    
    def get_latest_state(self) -> Optional[Dict[str, Any]]:
        """Get the most recent simulation state."""
        if not self.is_connected():
            return None
            
        try:
            latest = self.redis.get('simulation:latest')
            if latest:
                return json.loads(latest)
            return None
        except Exception as e:
            print(f"Error getting latest state: {e}")
            return None
    
    def store_simulation_metrics(self, metrics: Dict[str, Any]) -> bool:
        """
        Store aggregated simulation metrics.
        
        Args:
            metrics: Dictionary of metrics to store
            
        Returns:
            bool: Success status
        """
        if not self.is_connected():
            return False
            
        try:
            timestamp = int(time.time())
            
            # Store each metric as a time-series
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.redis.zadd(f'metrics:{metric_name}', {value: timestamp})
                    
                    # Keep only last 24 hours
                    yesterday = timestamp - 86400
                    self.redis.zremrangebyscore(f'metrics:{metric_name}', 0, yesterday)
            
            return True
            
        except Exception as e:
            print(f"Error storing metrics: {e}")
            return False
    
    def get_metric_history(self, metric_name: str, hours: int = 1) -> List[tuple]:
        """
        Get historical data for a specific metric.
        
        Args:
            metric_name: Name of the metric
            hours: Number of hours of history to retrieve
            
        Returns:
            List of (value, timestamp) tuples
        """
        if not self.is_connected():
            return []
            
        try:
            start_time = int(time.time()) - (hours * 3600)
            end_time = int(time.time())
            
            data = self.redis.zrangebyscore(
                f'metrics:{metric_name}',
                start_time,
                end_time,
                withscores=True
            )
            
            return [(float(value), int(score)) for value, score in data]
            
        except Exception as e:
            print(f"Error getting metric history: {e}")
            return []
    
    def start_simulation_session(self, scenario_name: str, config: Dict[str, Any]) -> str:
        """
        Start a new simulation session and store its metadata.
        
        Args:
            scenario_name: Name of the scenario being run
            config: Simulation configuration
            
        Returns:
            str: Session ID
        """
        if not self.is_connected():
            return ""
            
        try:
            session_id = f"sim_{int(time.time() * 1000)}"
            session_data = {
                'session_id': session_id,
                'scenario_name': scenario_name,
                'config': config,
                'start_time': time.time(),
                'status': 'running'
            }
            
            # Store session metadata
            self.redis.setex(f'session:{session_id}', 86400, json.dumps(session_data, cls=SimulationJSONEncoder))  # 24 hour TTL
            self.redis.set('simulation:current_session', session_id)
            
            # Clear previous simulation data
            self.redis.delete('simulation:timeseries')
            for key in self.redis.keys('metrics:*'):
                self.redis.delete(key)
            
            print(f"📝 Started simulation session: {session_id}")
            return session_id
            
        except Exception as e:
            print(f"Error starting simulation session: {e}")
            return ""
    
    def end_simulation_session(self) -> bool:
        """
        End the current simulation session.
        
        Returns:
            bool: Success status
        """
        if not self.is_connected():
            return False
            
        try:
            session_id = self.redis.get('simulation:current_session')
            if session_id:
                # Update session status
                session_data_raw = self.redis.get(f'session:{session_id}')
                if session_data_raw:
                    session_data = json.loads(session_data_raw)
                    session_data['status'] = 'completed'
                    session_data['end_time'] = time.time()
                    session_data['duration'] = session_data['end_time'] - session_data['start_time']
                    
                    # Store final session data
                    self.redis.setex(f'session:{session_id}', 86400, json.dumps(session_data, cls=SimulationJSONEncoder))
                
                # Clear current session
                self.redis.delete('simulation:current_session')
                print(f"📝 Ended simulation session: {session_id}")
            
            return True
            
        except Exception as e:
            print(f"Error ending simulation session: {e}")
            return False
    
    def get_current_session_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current simulation session."""
        if not self.is_connected():
            return None
            
        try:
            session_id = self.redis.get('simulation:current_session')
            if session_id:
                session_data_raw = self.redis.get(f'session:{session_id}')
                if session_data_raw:
                    return json.loads(session_data_raw)
            return None
        except Exception as e:
            print(f"Error getting session info: {e}")
            return None
        """Clear all simulation data from Redis."""
        if not self.is_connected():
            return False
            
        try:
            # Delete simulation keys
            keys_to_delete = []
            for pattern in ['simulation:*', 'metrics:*']:
                keys_to_delete.extend(self.redis.keys(pattern))
            
            if keys_to_delete:
                self.redis.delete(*keys_to_delete)
            
            return True
            
        except Exception as e:
            print(f"Error clearing Redis data: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis usage statistics."""
        if not self.is_connected():
            return {}
            
        try:
            info = self.redis.info()
            broadcast_count = self.redis.zcard('simulation:broadcast')
            
            return {
                'connected': True,
                'redis_version': info.get('redis_version'),
                'used_memory_human': info.get('used_memory_human'),
                'connected_clients': info.get('connected_clients'),
                'total_commands_processed': info.get('total_commands_processed'),
                'broadcast_entries': broadcast_count,
                'uptime_in_seconds': info.get('uptime_in_seconds')
            }
            
        except Exception as e:
            return {'connected': False, 'error': str(e)}


# Global Redis client instance
redis_client = RedisClient()
