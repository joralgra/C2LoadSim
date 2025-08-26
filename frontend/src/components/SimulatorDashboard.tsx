import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Card,
  CardContent,
  LinearProgress,
  Chip,
  Stack,
  Divider,
  Alert,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  ButtonGroup,
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Refresh,
  Assessment,
  WorkOutline,
  Queue,
  Wifi,
  WifiOff,
  Pause,
  PlayCircleOutline,
  SkipNext,
  FastForward,
} from '@mui/icons-material';

import {
  simulatorAPI,
  SimulationStatus,
  QueueStats,
  WorkerStats,
} from '../api/simulatorAPI';

import { useWebSocket } from '../contexts/WebSocketProvider';

interface DashboardState {
  simulationStatus: SimulationStatus & { paused?: boolean };
  queueStats: QueueStats | null;
  workerStats: WorkerStats[];
  loading: boolean;
  error: string | null;
}

interface StepControlState {
  stepTime: number;
  stepUnit: string;
}

const SimulatorDashboard: React.FC = () => {
  const { isConnected: wsConnected, latestData, connectionError: wsError } = useWebSocket();
  
  const [state, setState] = useState<DashboardState>({
    simulationStatus: {
      running: false,
      paused: false,
      simulation_time: 0,
      total_jobs_created: 0,
      queue_length: 0,
      active_workers: 0,
    },
    queueStats: null,
    workerStats: [],
    loading: false,
    error: null,
  });

  const [stepControl, setStepControl] = useState<StepControlState>({
    stepTime: 1,
    stepUnit: 'minutes'
  });

  const [lastUpdate, setLastUpdate] = useState<number | null>(null);

  // Update state when WebSocket data arrives
  useEffect(() => {
    if (latestData) {
      setState(prev => ({
        ...prev,
        simulationStatus: latestData.status,
        queueStats: latestData.queue_stats,
        workerStats: latestData.worker_stats,
        error: null,
      }));
      setLastUpdate(latestData.timestamp);
    }
  }, [latestData]);

  // Handle WebSocket errors
  useEffect(() => {
    if (wsError) {
      setState(prev => ({
        ...prev,
        error: wsError,
      }));
    }
  }, [wsError]);

  // Fallback: Fetch data manually if WebSocket is not connected
  const fetchData = async () => {
    if (wsConnected) return; // Don't fetch if WebSocket is working
    
    try {
      const [status, queueStats, workerData] = await Promise.all([
        simulatorAPI.getSimulationStatus(),
        simulatorAPI.getQueueStats(),
        simulatorAPI.getWorkerStats(),
      ]);

      setState(prev => ({
        ...prev,
        simulationStatus: status,
        queueStats,
        workerStats: workerData.workers,
        error: null,
      }));
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to fetch data',
      }));
    }
  };

  useEffect(() => {
    // Only set up polling if WebSocket is not connected
    if (!wsConnected) {
      fetchData(); // Initial fetch
      const interval = setInterval(fetchData, 5000); // Poll every 5 seconds as fallback
      return () => clearInterval(interval);
    }
  }, [wsConnected]);

  const handleStartSimulation = async () => {
    setState(prev => ({ ...prev, loading: true }));
    
    try {
      // Basic simulation configuration
      const config = {
        scenario: {
          name: "Default Simulation",
          description: "Basic workload simulation",
          duration: 300, // 5 minutes
          num_workers: 4,
          worker_config: {
            processing_speed: 1.0,
            failure_rate: 0.01,
            efficiency_variance: 0.1,
            max_concurrent_jobs: 1,
          },
          job_generation: {
            job_size: {
              distribution: "lognormal",
              params: { mean_log: 0.0, std_log: 0.5 }
            },
            job_duration: {
              distribution: "gamma",
              params: { alpha: 2.0, beta: 1.0 }
            },
            inter_arrival_time: {
              distribution: "exponential",
              params: { lambda: 1.0 }
            },
            job_types: {
              milp: 0.3,
              heuristic: 0.4,
              ml: 0.2,
              mixed: 0.1
            }
          }
        }
      };

      await simulatorAPI.startSimulation(config);
      fetchData(); // Immediately update data
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to start simulation',
      }));
    } finally {
      setState(prev => ({ ...prev, loading: false }));
    }
  };

  const handleStopSimulation = async () => {
    setState(prev => ({ ...prev, loading: true }));
    
    try {
      await simulatorAPI.stopSimulation();
      fetchData(); // Immediately update data
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to stop simulation',
      }));
    } finally {
      setState(prev => ({ ...prev, loading: false }));
    }
  };

  const handleResetSimulation = async () => {
    setState(prev => ({ ...prev, loading: true }));
    
    try {
      await simulatorAPI.resetSimulation();
      fetchData(); // Immediately update data
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to reset simulation',
      }));
    } finally {
      setState(prev => ({ ...prev, loading: false }));
    }
  };

  const handlePauseSimulation = async () => {
    setState(prev => ({ ...prev, loading: true }));
    
    try {
      await simulatorAPI.pauseSimulation();
      fetchData(); // Immediately update data
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to pause simulation',
      }));
    } finally {
      setState(prev => ({ ...prev, loading: false }));
    }
  };

  const handleResumeSimulation = async () => {
    setState(prev => ({ ...prev, loading: true }));
    
    try {
      await simulatorAPI.resumeSimulation();
      fetchData(); // Immediately update data
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to resume simulation',
      }));
    } finally {
      setState(prev => ({ ...prev, loading: false }));
    }
  };

  const handleStepOnce = async () => {
    setState(prev => ({ ...prev, loading: true }));
    
    try {
      await simulatorAPI.stepSimulation();
      fetchData(); // Immediately update data
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to step simulation',
      }));
    } finally {
      setState(prev => ({ ...prev, loading: false }));
    }
  };

  const handleStepTime = async () => {
    setState(prev => ({ ...prev, loading: true }));
    
    try {
      await simulatorAPI.stepSimulationTime(stepControl.stepTime, stepControl.stepUnit);
      fetchData(); // Immediately update data
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to step simulation time',
      }));
    } finally {
      setState(prev => ({ ...prev, loading: false }));
    }
  };

  const formatTime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    
    if (hours > 0) {
      return `${hours}h ${minutes}m ${remainingSeconds}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${remainingSeconds}s`;
    } else {
      return `${remainingSeconds}s`;
    }
  };

  return (
    <Box>
      {/* Control Panel */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Simulation Control
        </Typography>
        
        {/* Primary Controls */}
        <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap" sx={{ mb: 2 }}>
          <Button
            variant="contained"
            color="primary"
            startIcon={<PlayArrow />}
            onClick={handleStartSimulation}
            disabled={state.loading || state.simulationStatus.running}
          >
            Start Simulation
          </Button>
          
          <Button
            variant="contained"
            color="error"
            startIcon={<Stop />}
            onClick={handleStopSimulation}
            disabled={state.loading || !state.simulationStatus.running}
          >
            Stop Simulation
          </Button>
          
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={handleResetSimulation}
            disabled={state.loading || state.simulationStatus.running}
          >
            Reset
          </Button>

          <Divider orientation="vertical" flexItem />

          {/* Pause/Resume Controls */}
          <ButtonGroup disabled={state.loading || !state.simulationStatus.running}>
            <Button
              variant={state.simulationStatus.paused ? "outlined" : "contained"}
              color="warning"
              startIcon={<Pause />}
              onClick={handlePauseSimulation}
              disabled={state.simulationStatus.paused}
            >
              Pause
            </Button>
            
            <Button
              variant={!state.simulationStatus.paused ? "outlined" : "contained"}
              color="success"
              startIcon={<PlayCircleOutline />}
              onClick={handleResumeSimulation}
              disabled={!state.simulationStatus.paused}
            >
              Resume
            </Button>
          </ButtonGroup>
        </Stack>

        {/* Step Controls - Only show when paused */}
        {state.simulationStatus.running && state.simulationStatus.paused && (
          <Paper variant="outlined" sx={{ p: 2, mb: 2, backgroundColor: 'grey.50' }}>
            <Typography variant="h6" gutterBottom>
              Step Controls
            </Typography>
            
            <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
              <Button
                variant="outlined"
                startIcon={<SkipNext />}
                onClick={handleStepOnce}
                disabled={state.loading}
                size="small"
              >
                Step Once
              </Button>

              <Divider orientation="vertical" flexItem />

              <TextField
                label="Time Period"
                type="number"
                value={stepControl.stepTime}
                onChange={(e) => setStepControl(prev => ({ 
                  ...prev, 
                  stepTime: Math.max(0.1, parseFloat(e.target.value) || 0.1) 
                }))}
                size="small"
                sx={{ width: 120 }}
                inputProps={{ min: 0.1, step: 0.1 }}
              />

              <FormControl size="small" sx={{ minWidth: 120 }}>
                <InputLabel>Unit</InputLabel>
                <Select
                  value={stepControl.stepUnit}
                  label="Unit"
                  onChange={(e) => setStepControl(prev => ({ 
                    ...prev, 
                    stepUnit: e.target.value 
                  }))}
                >
                  <MenuItem value="seconds">Seconds</MenuItem>
                  <MenuItem value="minutes">Minutes</MenuItem>
                  <MenuItem value="hours">Hours</MenuItem>
                  <MenuItem value="days">Days</MenuItem>
                  <MenuItem value="weeks">Weeks</MenuItem>
                </Select>
              </FormControl>

              <Button
                variant="contained"
                color="info"
                startIcon={<FastForward />}
                onClick={handleStepTime}
                disabled={state.loading}
                size="small"
              >
                Step Forward
              </Button>
            </Stack>
          </Paper>
        )}

        {/* Status Indicators */}
        <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
          <Chip
            label={
              state.simulationStatus.running 
                ? (state.simulationStatus.paused ? 'Paused' : 'Running') 
                : 'Stopped'
            }
            color={
              state.simulationStatus.running 
                ? (state.simulationStatus.paused ? 'warning' : 'success') 
                : 'default'
            }
            icon={
              state.simulationStatus.running 
                ? (state.simulationStatus.paused ? <Pause /> : <PlayArrow />) 
                : <Stop />
            }
          />

          <Chip
            label={wsConnected ? 'Live Updates' : 'Polling Mode'}
            color={wsConnected ? 'success' : 'warning'}
            icon={wsConnected ? <Wifi /> : <WifiOff />}
            size="small"
          />

          {lastUpdate && (
            <Typography variant="caption" color="textSecondary">
              Last update: {new Date(lastUpdate * 1000).toLocaleTimeString()}
            </Typography>
          )}
        </Stack>

        {state.loading && (
          <Box sx={{ mt: 2 }}>
            <LinearProgress />
          </Box>
        )}
      </Paper>

      {/* Status Overview Cards */}
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mb: 3 }}>
        <Card sx={{ flex: '1 1 250px', minWidth: 250 }}>
          <CardContent>
            <Typography color="textSecondary" gutterBottom>
              Simulation Time
            </Typography>
            <Typography variant="h6">
              {formatTime(state.simulationStatus.simulation_time)}
            </Typography>
          </CardContent>
        </Card>

        <Card sx={{ flex: '1 1 250px', minWidth: 250 }}>
          <CardContent>
            <Typography color="textSecondary" gutterBottom>
              Jobs Created
            </Typography>
            <Typography variant="h6">
              {state.simulationStatus.total_jobs_created}
            </Typography>
          </CardContent>
        </Card>

        <Card sx={{ flex: '1 1 250px', minWidth: 250 }}>
          <CardContent>
            <Typography color="textSecondary" gutterBottom>
              Queue Length
            </Typography>
            <Typography variant="h6">
              {state.simulationStatus.queue_length}
            </Typography>
          </CardContent>
        </Card>

        <Card sx={{ flex: '1 1 250px', minWidth: 250 }}>
          <CardContent>
            <Typography color="textSecondary" gutterBottom>
              Active Workers
            </Typography>
            <Typography variant="h6">
              {state.simulationStatus.active_workers} / {state.workerStats.length}
            </Typography>
          </CardContent>
        </Card>
      </Box>

      {/* Detailed Statistics */}
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
        {/* Queue Statistics */}
        <Paper sx={{ p: 3, flex: '1 1 400px', minWidth: 400 }}>
          <Typography variant="h6" gutterBottom>
            <Queue sx={{ mr: 1, verticalAlign: 'middle' }} />
            Queue Statistics
          </Typography>
          
          {state.queueStats ? (
            <Stack spacing={2}>
              <Box>
                <Typography variant="body2" color="textSecondary">
                  Queue Utilization
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={state.queueStats.queue_utilization * 100} 
                  sx={{ mt: 1 }}
                />
                <Typography variant="caption">
                  {Math.round(state.queueStats.queue_utilization * 100)}%
                </Typography>
              </Box>
              
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Box>
                  <Typography variant="body2" color="textSecondary">Completed</Typography>
                  <Typography variant="h6">{state.queueStats.completed_jobs}</Typography>
                </Box>
                <Box>
                  <Typography variant="body2" color="textSecondary">Failed</Typography>
                  <Typography variant="h6">{state.queueStats.failed_jobs}</Typography>
                </Box>
                <Box>
                  <Typography variant="body2" color="textSecondary">Success Rate</Typography>
                  <Typography variant="h6">{Math.round(state.queueStats.success_rate * 100)}%</Typography>
                </Box>
              </Box>
            </Stack>
          ) : (
            <Typography color="textSecondary">No queue data available</Typography>
          )}
        </Paper>

        {/* Worker Statistics */}
        <Paper sx={{ p: 3, flex: '1 1 400px', minWidth: 400 }}>
          <Typography variant="h6" gutterBottom>
            <WorkOutline sx={{ mr: 1, verticalAlign: 'middle' }} />
            Worker Statistics
          </Typography>
          
          {state.workerStats.length > 0 ? (
            <Stack spacing={2}>
              {state.workerStats.map((worker, index) => (
                <Box key={worker.worker_id}>
                  <Stack direction="row" justifyContent="space-between" alignItems="center">
                    <Typography variant="body2">
                      Worker {index + 1}
                    </Typography>
                    <Chip 
                      label={worker.status} 
                      size="small"
                      color={worker.status === 'busy' ? 'primary' : 'default'}
                    />
                  </Stack>
                  <LinearProgress 
                    variant="determinate" 
                    value={worker.utilization * 100} 
                    sx={{ mt: 0.5 }}
                  />
                  <Stack direction="row" justifyContent="space-between" sx={{ mt: 0.5 }}>
                    <Typography variant="caption">
                      Utilization: {Math.round(worker.utilization * 100)}%
                    </Typography>
                    <Typography variant="caption">
                      Completed: {worker.completed_jobs}
                    </Typography>
                  </Stack>
                </Box>
              ))}
            </Stack>
          ) : (
            <Typography color="textSecondary">No worker data available</Typography>
          )}
        </Paper>
      </Box>

      {/* Error Display */}
      {state.error && (
        <Paper sx={{ p: 2, mt: 3, backgroundColor: 'error.dark' }}>
          <Typography color="error">
            Error: {state.error}
          </Typography>
        </Paper>
      )}
    </Box>
  );
};

export default SimulatorDashboard;
