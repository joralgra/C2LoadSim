/**
 * API handler for backend communication
 * 
 * Provides functions to interact with the C2LoadSim backend API
 */

import axios, { AxiosResponse } from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000/api';

// Types
export interface SimulationStatus {
  running: boolean;
  paused: boolean;
  simulation_time: number;
  total_jobs_created: number;
  queue_length: number;
  active_workers: number;
}

export interface QueueStats {
  queue_length: number;
  max_queue_size: number;
  queue_utilization: number;
  completed_jobs: number;
  failed_jobs: number;
  total_jobs_processed: number;
  success_rate: number;
  use_priority_queue: boolean;
}

export interface WorkerStats {
  worker_id: string;
  status: string;
  processing_speed: number;
  failure_rate: number;
  current_jobs: number;
  utilization: number;
  completed_jobs: number;
  failed_jobs: number;
  success_rate: number;
  average_processing_time: number;
  uptime_seconds: number;
}

export interface ScenarioConfig {
  name: string;
  description?: string;
  duration: number;
  num_workers?: number;
  queue_config?: any;
  worker_config?: {
    processing_speed?: number;
    failure_rate?: number;
    efficiency_variance?: number;
    max_concurrent_jobs?: number;
  };
  job_generation?: any;
}

export interface SimulationConfig {
  scenario: ScenarioConfig;
  log_directory?: string;
}

class SimulatorAPI {
  private axiosInstance = axios.create({
    baseURL: API_BASE_URL,
    timeout: 10000,
    headers: {
      'Content-Type': 'application/json',
    },
  });

  constructor() {
    // Add response interceptor for error handling
    this.axiosInstance.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error);
        throw error;
      }
    );
  }

  // Health check
  async healthCheck(): Promise<{ status: string; message: string }> {
    const response: AxiosResponse = await this.axiosInstance.get('/health');
    return response.data;
  }

  // Simulation control
  async startSimulation(config: SimulationConfig): Promise<SimulationStatus> {
    const response: AxiosResponse = await this.axiosInstance.post('/simulation/start', config);
    return response.data;
  }

  async getSimulationStatus(): Promise<SimulationStatus> {
    const response: AxiosResponse = await this.axiosInstance.get('/simulation/status');
    return response.data;
  }

  async stopSimulation(): Promise<{ message: string }> {
    const response: AxiosResponse = await this.axiosInstance.post('/simulation/stop');
    return response.data;
  }

  async resetSimulation(): Promise<{ message: string }> {
    const response: AxiosResponse = await this.axiosInstance.post('/simulation/reset');
    return response.data;
  }

  async pauseSimulation(): Promise<{ message: string }> {
    const response: AxiosResponse = await this.axiosInstance.post('/simulation/pause');
    return response.data;
  }

  async resumeSimulation(): Promise<{ message: string }> {
    const response: AxiosResponse = await this.axiosInstance.post('/simulation/resume');
    return response.data;
  }

  async stepSimulation(): Promise<any> {
    const response: AxiosResponse = await this.axiosInstance.post('/simulation/step');
    return response.data;
  }

  async stepSimulationTime(timePeriod: number, unit: string = 'seconds'): Promise<any> {
    const response: AxiosResponse = await this.axiosInstance.post('/simulation/step-time', {
      time_period: timePeriod,
      unit: unit
    });
    return response.data;
  }

  // Scenario management
  async uploadScenario(file: File): Promise<{ message: string; filename: string; scenario_name: string }> {
    const formData = new FormData();
    formData.append('file', file);
    
    const response: AxiosResponse = await this.axiosInstance.post('/scenarios/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  async listScenarios(): Promise<{ scenarios: Array<{ filename: string; name: string; description: string; duration: number }> }> {
    const response: AxiosResponse = await this.axiosInstance.get('/scenarios/list');
    return response.data;
  }

  async getScenario(filename: string): Promise<ScenarioConfig> {
    const response: AxiosResponse = await this.axiosInstance.get(`/scenarios/${filename}`);
    return response.data;
  }

  // Data retrieval
  async getLogSummary(): Promise<any> {
    const response: AxiosResponse = await this.axiosInstance.get('/data/logs/summary');
    return response.data;
  }

  async downloadJSONLogs(): Promise<Blob> {
    const response: AxiosResponse = await this.axiosInstance.get('/data/logs/json', {
      responseType: 'blob',
    });
    return response.data;
  }

  async downloadCSVLogs(): Promise<Blob> {
    const response: AxiosResponse = await this.axiosInstance.get('/data/logs/csv', {
      responseType: 'blob',
    });
    return response.data;
  }

  async getQueueStats(): Promise<QueueStats> {
    const response: AxiosResponse = await this.axiosInstance.get('/data/queue/stats');
    return response.data;
  }

  async getWorkerStats(): Promise<{ workers: WorkerStats[] }> {
    const response: AxiosResponse = await this.axiosInstance.get('/data/workers/stats');
    return response.data;
  }

  // Utility methods
  downloadFile(blob: Blob, filename: string): void {
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  }
}

// Export singleton instance
export const simulatorAPI = new SimulatorAPI();

export default simulatorAPI;
