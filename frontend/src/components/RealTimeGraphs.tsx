import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  Card,
  CardContent,
  Stack,
  Chip,
  Switch,
  FormControlLabel,
  Alert,
  Button,
  ButtonGroup,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Tooltip,
  IconButton,
} from '@mui/material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
} from 'recharts';
import {
  Timeline,
  TrendingUp,
  PeopleAlt,
  WorkOutline,
  Assessment,
  Refresh,
  Download,
  Fullscreen,
  ZoomIn,
  Settings,
} from '@mui/icons-material';

import { useWebSocket } from '../contexts/WebSocketProvider';
import simulatorAPI from '../api/simulatorAPI';

interface SimulationDataPoint {
  timestamp: number;
  simulation_time: number;
  total_jobs_created: number;
  jobs_in_queue: number;
  jobs_completed: number;
  jobs_failed: number;
  total_workers: number;
  active_workers: number;
  average_utilization: number;
  throughput: number;
  success_rate: number;
}

interface WorkerDataPoint {
  timestamp: number;
  simulation_time: number;
  worker_id: string;
  worker_index: number;
  status: string;
  utilization: number;
  completed_jobs: number;
  failed_jobs: number;
  current_jobs: number;
}

interface RealTimeGraphsProps {
  maxDataPoints?: number;
}

interface GraphSettings {
  timeRange: 'all' | '5m' | '15m' | '30m' | '1h';
  refreshRate: number; // seconds
  showGrid: boolean;
  showLegend: boolean;
  chartType: 'line' | 'area';
}

const RealTimeGraphs: React.FC<RealTimeGraphsProps> = ({ maxDataPoints = 2000 }) => {
  const { isConnected, latestData, clearGraphDataTrigger } = useWebSocket();
  const [simulationData, setSimulationData] = useState<SimulationDataPoint[]>([]);
  const [workerData, setWorkerData] = useState<WorkerDataPoint[]>([]);
  const [showRealTime, setShowRealTime] = useState(true);
  const [lastUpdateTime, setLastUpdateTime] = useState<number | null>(null);
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [dataSource, setDataSource] = useState<'none' | 'redis' | 'file' | 'realtime'>('none');
  
  const [settings, setSettings] = useState<GraphSettings>({
    timeRange: 'all',
    refreshRate: 1,
    showGrid: true,
    showLegend: true,
    chartType: 'line',
  });

  // Manual data refresh function
  // Function to manually load historical data from Redis/files
  const loadHistoricalData = useCallback(async () => {
    console.log('📜 Manually loading historical data...');
    setIsLoading(true);
    
    try {
      const response = await simulatorAPI.getSimulationHistory();
      console.log('📜 Raw API response:', response);
      console.log('📜 History source:', response.source);
      console.log('📜 Total history entries from backend:', response.count || 0);
      
      if (response.data && response.data.length > 0) {
        // Log a sample of the raw history data
        console.log('📜 First few history entries:', response.data.slice(0, 3));
        console.log('📜 Last few history entries:', response.data.slice(-3));
        
        // Convert log data to our format - load ALL available historical data
        const historicalSimData: SimulationDataPoint[] = [];
        const historicalWorkerData: WorkerDataPoint[] = [];

        // Use all history data
        response.data.forEach((entry: any) => {
          const timestamp = new Date(entry.timestamp).getTime();
          const simTime = entry.simulation_time || 0;

          // Add simulation data point
          historicalSimData.push({
            timestamp,
            simulation_time: simTime,
            total_jobs_created: entry.total_jobs_created || 0,
            jobs_in_queue: entry.queue_stats?.queue_length || 0,
            jobs_completed: entry.queue_stats?.completed_jobs || 0,
            jobs_failed: entry.queue_stats?.failed_jobs || 0,
            total_workers: entry.worker_stats?.total_workers || 0,
            active_workers: entry.worker_stats?.active_workers || 0,
            average_utilization: (entry.worker_stats?.average_utilization || 0) * 100,
            throughput: entry.performance_metrics?.throughput || 0,
            success_rate: (entry.performance_metrics?.success_rate || 0) * 100,
          });

          // Add worker data points
          if (entry.worker_stats?.individual_utilizations) {
            entry.worker_stats.individual_utilizations.forEach((util: number, index: number) => {
              historicalWorkerData.push({
                timestamp,
                simulation_time: simTime,
                worker_id: `worker_${index}`,
                worker_index: index + 1,
                status: 'active',
                utilization: util * 100,
                completed_jobs: 0, // Default values for required properties
                failed_jobs: 0,
                current_jobs: 0,
              });
            });
          }
        });

        // Update state with historical data
        setSimulationData(historicalSimData);
        setWorkerData(historicalWorkerData);
        setHistoryLoaded(true);
        setDataSource(response.source as 'redis' | 'file');
        
        console.log('✅ Successfully loaded historical data:', {
          simulationPoints: historicalSimData.length,
          workerPoints: historicalWorkerData.length,
          source: response.source,
          timeRange: historicalSimData.length > 0 ? {
            start: new Date(historicalSimData[0].timestamp),
            end: new Date(historicalSimData[historicalSimData.length - 1].timestamp)
          } : null
        });
      } else {
        console.log('📜 No historical data available');
        // Clear existing data if no historical data is found
        setSimulationData([]);
        setWorkerData([]);
        setHistoryLoaded(true);
      }
    } catch (error) {
      console.error('❌ Error loading historical data:', error);
      setHistoryLoaded(true); // Set to true to prevent infinite retry
    } finally {
      setIsLoading(false);
    }
  }, []);

  const refreshData = useCallback(async () => {
    if (!isConnected) return;
    
    setIsLoading(true);
    try {
      const response = await simulatorAPI.getSimulationHistory();
      if (response.data && response.data.length > 0) {
        // Process the latest data point
        const latest = response.data[response.data.length - 1];
        const timestamp = new Date(latest.timestamp).getTime();
        const simTime = latest.simulation_time || 0;

        const newSimPoint: SimulationDataPoint = {
          timestamp,
          simulation_time: simTime,
          total_jobs_created: latest.total_jobs_created || 0,
          jobs_in_queue: latest.queue_stats?.queue_length || 0,
          jobs_completed: latest.queue_stats?.completed_jobs || 0,
          jobs_failed: latest.queue_stats?.failed_jobs || 0,
          total_workers: latest.worker_stats?.total_workers || 0,
          active_workers: latest.worker_stats?.active_workers || 0,
          average_utilization: (latest.worker_stats?.average_utilization || 0) * 100,
          throughput: latest.performance_metrics?.throughput || 0,
          success_rate: (latest.performance_metrics?.success_rate || 0) * 100,
        };

        setSimulationData(prev => {
          // Only add if it's newer than the last data point
          if (prev.length === 0 || timestamp > prev[prev.length - 1].timestamp) {
            return [...prev, newSimPoint].slice(-maxDataPoints);
          }
          return prev;
        });
        
        setLastUpdateTime(timestamp);
      }
    } catch (error) {
      console.error('Failed to refresh data:', error);
    } finally {
      setIsLoading(false);
    }
  }, [isConnected, maxDataPoints]);

  // Filter data based on time range
  const getFilteredData = useCallback((data: SimulationDataPoint[]) => {
    if (settings.timeRange === 'all' || data.length === 0) return data;
    
    // Use the latest data point timestamp as reference instead of current time
    // This prevents data from disappearing when simulation is stopped
    const latestTimestamp = Math.max(...data.map(point => point.timestamp));
    const timeRanges = {
      '5m': 5 * 60 * 1000,
      '15m': 15 * 60 * 1000, 
      '30m': 30 * 60 * 1000,
      '1h': 60 * 60 * 1000,
    };
    
    const cutoff = latestTimestamp - timeRanges[settings.timeRange];
    return data.filter(point => point.timestamp >= cutoff);
  }, [settings.timeRange]);

  // Filter worker data based on time range  
  const getFilteredWorkerData = useCallback((data: WorkerDataPoint[]) => {
    if (settings.timeRange === 'all' || data.length === 0) return data;
    
    // Use the latest worker data point timestamp as reference
    const latestTimestamp = Math.max(...data.map(point => point.timestamp || 0));
    const timeRanges = {
      '5m': 5 * 60 * 1000,
      '15m': 15 * 60 * 1000, 
      '30m': 30 * 60 * 1000,
      '1h': 60 * 60 * 1000,
    };
    
    const cutoff = latestTimestamp - timeRanges[settings.timeRange];
    return data.filter(point => (point.timestamp || 0) >= cutoff);
  }, [settings.timeRange]);

  // Load historical data when component mounts
  useEffect(() => {
    const loadHistoricalData = async () => {
      console.log('📜 Loading historical data... Current data points:', simulationData.length);
      try {
        const response = await simulatorAPI.getSimulationHistory();
        console.log('📜 Raw API response:', response);
        console.log('📜 History source:', response.source);
        console.log('📜 Total history entries from backend:', response.count || 0);
        
        if (response.data && response.data.length > 0) {
          // Log a sample of the raw history data
          console.log('📜 First few history entries:', response.data.slice(0, 3));
          console.log('📜 Last few history entries:', response.data.slice(-3));
          
          // Convert log data to our format - load ALL available historical data
          const historicalSimData: SimulationDataPoint[] = [];
          const historicalWorkerData: WorkerDataPoint[] = [];

          // Use all history data, not just the last maxDataPoints entries
          response.data.forEach((entry: any) => {
            const timestamp = new Date(entry.timestamp).getTime();
            const simTime = entry.simulation_time || 0;

            // Add simulation data point
            historicalSimData.push({
              timestamp,
              simulation_time: simTime,
              total_jobs_created: entry.total_jobs_created || 0,
              jobs_in_queue: entry.queue_stats?.queue_length || 0,
              jobs_completed: entry.queue_stats?.completed_jobs || 0,
              jobs_failed: entry.queue_stats?.failed_jobs || 0,
              total_workers: entry.worker_stats?.total_workers || 0,
              active_workers: entry.worker_stats?.active_workers || 0,
              average_utilization: (entry.worker_stats?.average_utilization || 0) * 100,
              throughput: entry.performance_metrics?.throughput || 0,
              success_rate: (entry.performance_metrics?.success_rate || 0) * 100,
            });

            // Add worker data points
            if (entry.worker_stats?.individual_utilizations) {
              entry.worker_stats.individual_utilizations.forEach((util: number, index: number) => {
                historicalWorkerData.push({
                  timestamp,
                  simulation_time: simTime,
                  worker_id: `worker_${index}`,
                  worker_index: index + 1,
                  status: 'active',
                  utilization: util * 100,
                  completed_jobs: 0,
                  failed_jobs: 0,
                  current_jobs: 0,
                });
              });
            }
          });

          setSimulationData(prev => {
            // Only replace data if we don't have real-time data already
            if (prev.length === 0) {
              console.log('📊 Setting historical data:', historicalSimData.length, 'points');
              console.log('📊 Time range:', historicalSimData.length > 0 ? 
                `${(historicalSimData[0].simulation_time/60).toFixed(1)} - ${(historicalSimData[historicalSimData.length-1].simulation_time/60).toFixed(1)} minutes` : 
                'No data');
              return historicalSimData;
            }
            console.log('📊 Keeping existing real-time data:', prev.length, 'points');
            return prev; // Keep existing real-time data
          });
          setWorkerData(prev => {
            // Only replace data if we don't have real-time data already  
            if (prev.length === 0) {
              console.log('👷 Setting historical worker data:', historicalWorkerData.length, 'points');
              return historicalWorkerData;
            }
            console.log('👷 Keeping existing worker data:', prev.length, 'points');
            return prev; // Keep existing real-time data
          });
        }
        setHistoryLoaded(true);
      } catch (error) {
        console.error('Failed to load historical data:', error);
        setHistoryLoaded(true); // Still mark as loaded to avoid infinite retries
      }
    };

    if (!historyLoaded) {
      loadHistoricalData();
    }
  }, [maxDataPoints, historyLoaded]);

  // Process incoming WebSocket data
  useEffect(() => {
    if (latestData && showRealTime && isConnected && historyLoaded) {
      try {
        const timestamp = Date.now();
        const simTime = latestData.status?.simulation_time || 0;

        // Safely process worker stats
        const workerStats = Array.isArray(latestData.worker_stats) ? latestData.worker_stats : [];
        const avgUtilization = workerStats.length > 0 
          ? workerStats.reduce((sum: number, w: any) => sum + (w?.utilization || 0), 0) / workerStats.length * 100
          : 0;

        // Add simulation data point
        const newSimPoint: SimulationDataPoint = {
          timestamp,
          simulation_time: simTime,
          total_jobs_created: latestData.status?.total_jobs_created || 0,
          jobs_in_queue: latestData.status?.queue_length || 0,
          jobs_completed: latestData.queue_stats?.completed_jobs || 0,
          jobs_failed: latestData.queue_stats?.failed_jobs || 0,
          total_workers: workerStats.length,
          active_workers: latestData.status?.active_workers || 0,
          average_utilization: avgUtilization,
          throughput: (latestData.queue_stats?.total_jobs_processed || 0) / Math.max(simTime, 0.001),
          success_rate: (latestData.queue_stats?.success_rate || 0) * 100,
        };

        setSimulationData(prev => {
          const updated = [...prev, newSimPoint];
          return updated.slice(-maxDataPoints);
        });

        // Add worker data points
        if (workerStats.length > 0) {
          const workerPoints = workerStats.map((worker: any, index: number) => ({
            timestamp,
            simulation_time: simTime,
            worker_id: worker?.worker_id || `worker_${index}`,
            worker_index: index + 1,
            status: worker?.status || 'idle',
            utilization: (worker?.utilization || 0) * 100,
            completed_jobs: worker?.completed_jobs || 0,
            failed_jobs: worker?.failed_jobs || 0,
            current_jobs: worker?.current_jobs || 0,
          }));

          setWorkerData(prev => {
            const updated = [...prev, ...workerPoints];
            return updated.slice(-maxDataPoints * Math.max(workerStats.length, 4));
          });
        }

        setLastUpdateTime(timestamp);
      } catch (error) {
        console.error('Error processing WebSocket data:', error);
      }
    }
  }, [latestData, showRealTime, isConnected, maxDataPoints, historyLoaded]);

  // Clear graph data when triggered from dashboard
  useEffect(() => {
    if (clearGraphDataTrigger && clearGraphDataTrigger > 0) {
      console.log('🔄 Clearing graph data triggered at:', new Date().toISOString());
      console.log('📊 Data before clearing - simulation points:', simulationData.length, 'worker points:', workerData.length);
      setSimulationData([]);
      setWorkerData([]);
      setLastUpdateTime(null);
      // Note: Don't reset historyLoaded here to avoid unwanted historical data reload
    }
  }, [clearGraphDataTrigger, simulationData.length, workerData.length]);

  // Memoized chart data processing with filtering and data point sampling
  // Process data for charts with memoization to reduce re-renders
  const chartData = useMemo(() => {
    console.log('🔄 Reprocessing chart data - simulation points:', simulationData.length, 'worker points:', workerData.length);
    try {
      const filteredData = getFilteredData(simulationData);
      const filteredWorkerData = getFilteredWorkerData(workerData);
      
      // Apply 30-minute interval sampling if we have a full day's worth of data (1440+ points)
      const shouldSampleData = filteredData.length >= 1440 && settings.timeRange === 'all';
      const sampleInterval = shouldSampleData ? 30 : 1; // Sample every 30th point for full dataset
      
      let sampledData = filteredData;
      if (shouldSampleData) {
        sampledData = filteredData.filter((_, index) => index % sampleInterval === 0 || index === filteredData.length - 1);
        console.log(`📊 Applied 30-minute sampling: ${filteredData.length} → ${sampledData.length} points`);
      }
      
      // Demand and queue data with proper type safety and sampling
      const demandData = sampledData
        .filter(point => point && typeof point.simulation_time === 'number')
        .map(point => ({
          time: parseFloat((point.simulation_time / 60).toFixed(1)), // Keep as number for chart
          'Jobs in Queue': Math.max(0, point.jobs_in_queue || 0),
          'Jobs Created': Math.max(0, point.total_jobs_created || 0),
          'Jobs Completed': Math.max(0, point.jobs_completed || 0),
          'Jobs Failed': Math.max(0, point.jobs_failed || 0),
        }));

      // Debug: Log the actual data being processed
      if (demandData.length > 0) {
        console.log('📈 Demand data sample:', {
          totalPoints: demandData.length,
          sampledFromOriginal: filteredData.length,
          sampleInterval: sampleInterval,
          firstPoint: demandData[0],
          lastPoint: demandData[demandData.length - 1],
          timeRange: `${demandData[0].time} - ${demandData[demandData.length - 1].time} minutes`
        });
      }

      // Performance data with type safety and sampling
      const performanceData = sampledData
        .filter(point => point && typeof point.simulation_time === 'number')
        .map(point => ({
          time: parseFloat((point.simulation_time / 60).toFixed(1)),
          'Worker Utilization (%)': Math.max(0, Math.min(100, point.average_utilization || 0)),
          'Throughput (jobs/min)': Math.max(0, (point.throughput || 0) * 60),
          'Success Rate (%)': Math.max(0, Math.min(100, point.success_rate || 0)),
        }));

      // Worker utilization data (latest snapshot) with safety checks
      const latestTimestamp = filteredWorkerData.length > 0 
        ? Math.max(...filteredWorkerData.map(w => w.timestamp || 0), 0)
        : 0;
      const latestWorkerData = filteredWorkerData
        .filter(w => w && w.timestamp === latestTimestamp)
        .map(w => ({
          worker: `Worker ${w.worker_index || 1}`,
          utilization: Math.max(0, Math.min(100, w.utilization || 0)),
          completed: Math.max(0, w.completed_jobs || 0),
          current: Math.max(0, w.current_jobs || 0),
        }));

      // Worker workload over time (for line chart) with proper validation and sampling
      const workerTimeData = sampledData
        .filter(point => point && typeof point.simulation_time === 'number')
        .map(point => {
          const timeKey = parseFloat((point.simulation_time / 60).toFixed(1));
          const workers = filteredWorkerData.filter(w => 
            w && Math.abs((w.simulation_time || 0) - point.simulation_time) < 1
          );
          
          const result: any = { time: timeKey };
          workers.forEach(worker => {
            if (worker && typeof worker.worker_index === 'number') {
              const workerKey = `Worker ${worker.worker_index}`;
              result[workerKey] = Math.max(0, Math.min(100, worker.utilization || 0));
            }
          });
          return result;
        });

      return {
        demand: demandData,
        performance: performanceData,
        currentWorkers: latestWorkerData,
        workerTimeline: workerTimeData,
        isFullDataset: shouldSampleData,
        originalDataPoints: filteredData.length,
        displayedDataPoints: sampledData.length,
        sampleInterval: sampleInterval,
      };
    } catch (error) {
      console.error('Error processing chart data:', error);
      return {
        demand: [],
        performance: [],
        currentWorkers: [],
        workerTimeline: [],
        isFullDataset: false,
        originalDataPoints: 0,
        displayedDataPoints: 0,
        sampleInterval: 1,
      };
    }
  }, [simulationData, workerData, getFilteredData, getFilteredWorkerData, settings.timeRange]);

  const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7c7c', '#8dd1e1', '#d084d0'];

  if (!isConnected) {
    return (
      <Alert severity="warning" sx={{ mb: 3 }}>
        Not connected to WebSocket. Real-time graphs are not available.
      </Alert>
    );
  }

  return (
    <Box>
      {/* Data Source Information */}
      {dataSource !== 'none' && (
        <Alert 
          severity={dataSource === 'redis' ? 'success' : dataSource === 'file' ? 'info' : 'warning'} 
          sx={{ mb: 2 }}
          action={
            <Button
              color="inherit"
              size="small"
              onClick={loadHistoricalData}
              disabled={isLoading}
            >
              Reload
            </Button>
          }
        >
          Data loaded from {dataSource === 'redis' ? 'Redis (high-performance storage)' : 
                          dataSource === 'file' ? 'file logs (backup storage)' : 
                          'real-time WebSocket connection'}
          {simulationData.length > 0 && ` • ${simulationData.length} data points`}
        </Alert>
      )}

      {/* Daily Simulation Information */}
      {simulationData.length >= 1440 && (
        <Alert severity="info" sx={{ mb: 2 }}>
          <strong>Daily Simulation Complete:</strong> This simulation ran for 24 hours with minute-by-minute data collection (1440 data points). 
          When viewing "All Data", graphs show 30-minute intervals for better visualization. Use time range filters for detailed views.
        </Alert>
      )}
      
      {/* Enhanced Control Panel */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Stack direction="row" justifyContent="space-between" alignItems="center" flexWrap="wrap" gap={2}>
          <Box>
            <Typography variant="h5" gutterBottom sx={{ mb: 0 }}>
              Live Analytics Dashboard
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Real-time simulation analytics with interactive charts and customizable views.
            </Typography>
          </Box>
          
          <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
            <Tooltip title="Refresh data manually">
              <IconButton
                onClick={refreshData}
                disabled={isLoading || !isConnected}
                color="primary"
              >
                <Refresh />
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Load persisted historical data from Redis/files">
              <Button
                variant="contained"
                startIcon={<Timeline />}
                onClick={loadHistoricalData}
                disabled={isLoading || !isConnected}
                size="small"
                color="secondary"
              >
                Load History
              </Button>
            </Tooltip>
            
            <Button
              variant="outlined"
              startIcon={<Download />}
              onClick={() => {
                // Export data functionality could be added here
                console.log('Export functionality - to be implemented');
              }}
              disabled={simulationData.length === 0}
              size="small"
            >
              Export Data
            </Button>
          </Stack>
        </Stack>

        {/* Enhanced Settings Row */}
        <Stack direction="row" spacing={3} alignItems="center" sx={{ mt: 3 }} flexWrap="wrap">
          <FormControlLabel
            control={
              <Switch
                checked={showRealTime}
                onChange={(e) => setShowRealTime(e.target.checked)}
                color="primary"
              />
            }
            label="Live Updates"
          />

          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Time Range</InputLabel>
            <Select
              value={settings.timeRange}
              label="Time Range"
              onChange={(e) => setSettings(prev => ({ 
                ...prev, 
                timeRange: e.target.value as GraphSettings['timeRange'] 
              }))}
            >
              <MenuItem value="all">All Data</MenuItem>
              <MenuItem value="5m">Last 5 min</MenuItem>
              <MenuItem value="15m">Last 15 min</MenuItem>
              <MenuItem value="30m">Last 30 min</MenuItem>
              <MenuItem value="1h">Last 1 hour</MenuItem>
            </Select>
          </FormControl>

          <ButtonGroup size="small">
            <Button
              variant={settings.chartType === 'line' ? 'contained' : 'outlined'}
              onClick={() => setSettings(prev => ({ ...prev, chartType: 'line' }))}
            >
              Lines
            </Button>
            <Button
              variant={settings.chartType === 'area' ? 'contained' : 'outlined'}
              onClick={() => setSettings(prev => ({ ...prev, chartType: 'area' }))}
            >
              Areas
            </Button>
          </ButtonGroup>

          <FormControlLabel
            control={
              <Switch
                checked={settings.showGrid}
                onChange={(e) => setSettings(prev => ({ ...prev, showGrid: e.target.checked }))}
                size="small"
              />
            }
            label="Grid"
          />

          <FormControlLabel
            control={
              <Switch
                checked={settings.showLegend}
                onChange={(e) => setSettings(prev => ({ ...prev, showLegend: e.target.checked }))}
                size="small"
              />
            }
            label="Legend"
          />
        </Stack>

        {/* Status Row */}
        <Stack direction="row" spacing={2} alignItems="center" sx={{ mt: 2 }} flexWrap="wrap">
          <Chip
            label={`${simulationData.length} data points`}
            variant="outlined"
            size="small"
            icon={<Assessment />}
          />
          
          {chartData.isFullDataset && (
            <Chip
              label={`30-minute intervals (${chartData.displayedDataPoints}/${chartData.originalDataPoints} points)`}
              variant="outlined"
              size="small"
              color="info"
              icon={<ZoomIn />}
            />
          )}
          
          {lastUpdateTime && (
            <Chip
              label={`Updated: ${new Date(lastUpdateTime).toLocaleTimeString()}`}
              variant="outlined"
              size="small"
              color={showRealTime ? "success" : "default"}
            />
          )}
          
          {isLoading && (
            <Chip
              label="Loading..."
              variant="outlined"
              size="small"
              color="info"
            />
          )}
        </Stack>
      </Paper>

      {/* Show message when no data */}
      {simulationData.length === 0 && (
        <Alert severity="info" sx={{ mb: 3 }}>
          No simulation data available yet. Start a simulation to see real-time graphs.
        </Alert>
      )}

      <Stack spacing={3}>
        {/* Demand and Queue Analysis */}
        <Card>
          <CardContent>
            <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 2 }}>
              <Timeline color="primary" />
              <Typography variant="h6">
                Demand and Queue Analysis
              </Typography>
            </Stack>
            
            {chartData.demand.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                {settings.chartType === 'line' ? (
                  <LineChart data={chartData.demand} margin={{ left: 20, right: 30, top: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={settings.showGrid ? undefined : "transparent"} />
                    <XAxis 
                      dataKey="time" 
                      type="number"
                      scale="linear"
                      domain={['dataMin', 'dataMax']}
                      label={{ value: 'Time (minutes)', position: 'insideBottom', offset: -10 }} 
                    />
                    <YAxis label={{ value: 'Number of Jobs', angle: -90, position: 'insideLeft' }} />
                    <RechartsTooltip />
                    {settings.showLegend && <Legend />}
                    <Line 
                      type="monotone" 
                      dataKey="Jobs Created" 
                      stroke="#ffc658" 
                      strokeWidth={2} 
                      dot={false} 
                      isAnimationActive={false}
                      connectNulls={false}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="Jobs in Queue" 
                      stroke="#8884d8" 
                      strokeWidth={2} 
                      dot={false} 
                      isAnimationActive={false}
                      connectNulls={false}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="Jobs Completed" 
                      stroke="#82ca9d" 
                      strokeWidth={2} 
                      dot={false} 
                      isAnimationActive={false}
                      connectNulls={false}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="Jobs Failed" 
                      stroke="#ff7c7c" 
                      strokeWidth={2} 
                      dot={false} 
                      isAnimationActive={false}
                      connectNulls={false}
                    />
                  </LineChart>
                ) : (
                  <AreaChart data={chartData.demand} margin={{ left: 20, right: 30, top: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={settings.showGrid ? undefined : "transparent"} />
                    <XAxis 
                      dataKey="time" 
                      type="number"
                      scale="linear"
                      domain={['dataMin', 'dataMax']}
                      label={{ value: 'Time (minutes)', position: 'insideBottom', offset: -10 }} 
                    />
                    <YAxis label={{ value: 'Number of Jobs', angle: -90, position: 'insideLeft' }} />
                    <RechartsTooltip />
                    {settings.showLegend && <Legend />}
                    <Area 
                      type="monotone" 
                      dataKey="Jobs Created" 
                      stackId="1" 
                      stroke="#ffc658" 
                      fill="#ffc658" 
                      fillOpacity={0.6} 
                      isAnimationActive={false}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="Jobs in Queue" 
                      stackId="1" 
                      stroke="#8884d8" 
                      fill="#8884d8" 
                      fillOpacity={0.6} 
                      isAnimationActive={false}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="Jobs Completed" 
                      stackId="1" 
                      stroke="#82ca9d" 
                      fill="#82ca9d" 
                      fillOpacity={0.6} 
                      isAnimationActive={false}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="Jobs Failed" 
                      stackId="1" 
                      stroke="#ff7c7c" 
                      fill="#ff7c7c" 
                      fillOpacity={0.6} 
                      isAnimationActive={false}
                    />
                  </AreaChart>
                )}
              </ResponsiveContainer>
            ) : (
              <Box sx={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'text.secondary' }}>
                <Typography>No data available - start a simulation to see real-time updates</Typography>
              </Box>
            )}
          </CardContent>
        </Card>

        {/* Performance Metrics */}
        <Card>
          <CardContent>
            <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 2 }}>
              <TrendingUp color="secondary" />
              <Typography variant="h6">
                Performance Metrics
              </Typography>
            </Stack>
            
            <ResponsiveContainer width="100%" height={300}>
              {settings.chartType === 'line' ? (
                <LineChart data={chartData.performance} margin={{ left: 20, right: 30, top: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={settings.showGrid ? undefined : "transparent"} />
                  <XAxis 
                    dataKey="time" 
                    type="number"
                    scale="linear"
                    domain={['dataMin', 'dataMax']}
                    label={{ value: 'Time (minutes)', position: 'insideBottom', offset: -10 }} 
                  />
                  <YAxis label={{ value: 'Percentage / Rate', angle: -90, position: 'insideLeft' }} />
                  <RechartsTooltip />
                  {settings.showLegend && <Legend />}
                  {chartData.performance.length > 0 && (
                    <>
                      <Line 
                        type="monotone" 
                        dataKey="Worker Utilization (%)" 
                        stroke="#8884d8" 
                        strokeWidth={2} 
                        dot={false} 
                        isAnimationActive={false}
                        connectNulls={false}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="Success Rate (%)" 
                        stroke="#82ca9d" 
                        strokeWidth={2} 
                        dot={false} 
                        isAnimationActive={false}
                        connectNulls={false}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="Throughput (jobs/min)" 
                        stroke="#ffc658" 
                        strokeWidth={2} 
                        dot={false} 
                        isAnimationActive={false}
                        connectNulls={false}
                      />
                    </>
                  )}
                </LineChart>
              ) : (
                <AreaChart data={chartData.performance} margin={{ left: 20, right: 30, top: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={settings.showGrid ? undefined : "transparent"} />
                  <XAxis 
                    dataKey="time" 
                    type="number"
                    scale="linear"
                    domain={['dataMin', 'dataMax']}
                    label={{ value: 'Time (minutes)', position: 'insideBottom', offset: -10 }} 
                  />
                  <YAxis label={{ value: 'Percentage / Rate', angle: -90, position: 'insideLeft' }} />
                  <RechartsTooltip />
                  {settings.showLegend && <Legend />}
                  {chartData.performance.length > 0 && (
                    <>
                      <Area 
                        type="monotone" 
                        dataKey="Worker Utilization (%)" 
                        stackId="1" 
                        stroke="#8884d8" 
                        fill="#8884d8" 
                        fillOpacity={0.6} 
                        isAnimationActive={false}
                      />
                      <Area 
                        type="monotone" 
                        dataKey="Success Rate (%)" 
                        stackId="1" 
                        stroke="#82ca9d" 
                        fill="#82ca9d" 
                        fillOpacity={0.6} 
                        isAnimationActive={false}
                      />
                      <Area 
                        type="monotone" 
                        dataKey="Throughput (jobs/min)" 
                        stackId="1" 
                        stroke="#ffc658" 
                        fill="#ffc658" 
                        fillOpacity={0.6} 
                        isAnimationActive={false}
                      />
                    </>
                  )}
                </AreaChart>
              )}
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Current Worker Status */}
        {chartData.currentWorkers.length > 0 && (
          <Card>
            <CardContent>
              <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 2 }}>
                <PeopleAlt color="success" />
                <Typography variant="h6">
                  Current Worker Utilization
                </Typography>
              </Stack>
              
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={chartData.currentWorkers} margin={{ left: 20, right: 30, top: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={settings.showGrid ? undefined : "transparent"} />
                  <XAxis dataKey="worker" label={{ value: 'Worker', position: 'insideBottom', offset: -10 }} />
                  <YAxis label={{ value: 'Utilization (%)', angle: -90, position: 'insideLeft' }} />
                  <RechartsTooltip />
                  <Bar dataKey="utilization" fill="#8884d8" isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}

        {/* Worker Workload Timeline */}
        {chartData.workerTimeline.length > 0 && (
          <Card>
            <CardContent>
              <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 2 }}>
                <WorkOutline color="info" />
                <Typography variant="h6">
                  Worker Workload Timeline
                </Typography>
              </Stack>
              
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData.workerTimeline} margin={{ left: 20, right: 30, top: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={settings.showGrid ? undefined : "transparent"} />
                  <XAxis 
                    dataKey="time" 
                    type="number" 
                    scale="linear"
                    domain={['dataMin', 'dataMax']}
                    label={{ value: 'Time (minutes)', position: 'insideBottom', offset: -10 }} 
                  />
                  <YAxis label={{ value: 'Utilization (%)', angle: -90, position: 'insideLeft' }} domain={[0, 100]} />
                  <RechartsTooltip />
                  {settings.showLegend && <Legend />}
                  {(() => {
                    // Dynamically determine which workers have data
                    const workerKeys = new Set<string>();
                    chartData.workerTimeline.forEach(point => {
                      Object.keys(point).forEach(key => {
                        if (key.startsWith('Worker ') && typeof point[key] === 'number') {
                          workerKeys.add(key);
                        }
                      });
                    });
                    
                    return Array.from(workerKeys).slice(0, 8).map((workerKey, index) => (
                      <Line
                        key={workerKey}
                        type="monotone"
                        dataKey={workerKey}
                        stroke={colors[index % colors.length]}
                        strokeWidth={2}
                        dot={false}
                        connectNulls={false}
                        isAnimationActive={false}
                      />
                    ));
                  })()}
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}

        {/* Summary Statistics */}
        {simulationData.length > 0 && (
          <Card>
            <CardContent>
              <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 2 }}>
                <Assessment color="warning" />
                <Typography variant="h6">
                  Current Statistics
                </Typography>
              </Stack>
              
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                {simulationData.length > 0 && (() => {
                  const latest = simulationData[simulationData.length - 1];
                  return (
                    <>
                      <Chip label={`Simulation Time: ${(latest.simulation_time / 60).toFixed(1)} min`} variant="outlined" />
                      <Chip label={`Jobs Created: ${latest.total_jobs_created}`} variant="outlined" />
                      <Chip label={`Queue Length: ${latest.jobs_in_queue}`} variant="outlined" />
                      <Chip label={`Jobs Completed: ${latest.jobs_completed}`} variant="outlined" />
                      <Chip label={`Success Rate: ${latest.success_rate.toFixed(1)}%`} variant="outlined" />
                      <Chip label={`Active Workers: ${latest.active_workers}/${latest.total_workers}`} variant="outlined" />
                      <Chip label={`Avg Utilization: ${latest.average_utilization.toFixed(1)}%`} variant="outlined" />
                      <Chip label={`Throughput: ${(latest.throughput * 60).toFixed(1)} jobs/min`} variant="outlined" />
                    </>
                  );
                })()}
              </Box>
            </CardContent>
          </Card>
        )}
      </Stack>
    </Box>
  );
};

export default RealTimeGraphs;
