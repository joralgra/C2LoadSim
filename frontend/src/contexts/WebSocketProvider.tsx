import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { io, Socket } from 'socket.io-client';
import { SimulationStatus, QueueStats, WorkerStats } from '../api/simulatorAPI';

interface SocketData {
  status: SimulationStatus;
  queue_stats: QueueStats;
  worker_stats: WorkerStats[];
  timestamp: number;
}

interface WebSocketContextType {
  socket: Socket | null;
  isConnected: boolean;
  latestData: SocketData | null;
  connectionError: string | null;
  clearGraphData: () => void;
  clearGraphDataTrigger: number | null;
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (context === undefined) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};

interface WebSocketProviderProps {
  children: ReactNode;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children }) => {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [latestData, setLatestData] = useState<SocketData | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [clearGraphDataTrigger, setClearGraphDataTrigger] = useState<number | null>(null);

  const clearGraphData = () => {
    setClearGraphDataTrigger(Date.now());
  };

  useEffect(() => {
    const SOCKET_SERVER_URL = process.env.REACT_APP_API_BASE_URL?.replace('/api', '') || 'http://localhost:5000';
    
    console.log('Connecting to WebSocket server:', SOCKET_SERVER_URL);
    
    const newSocket = io(SOCKET_SERVER_URL, {
      transports: ['polling', 'websocket'],
      upgrade: true,
      rememberUpgrade: true
    });

    // Connection event handlers
    newSocket.on('connect', () => {
      console.log('WebSocket connected');
      setIsConnected(true);
      setConnectionError(null);
      
      // Join the simulation updates room
      newSocket.emit('join_simulation');
    });

    newSocket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
      setIsConnected(false);
      if (reason === 'io server disconnect') {
        // The disconnection was initiated by the server, reconnect manually
        newSocket.connect();
      }
    });

    newSocket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      setConnectionError(`Connection failed: ${error.message}`);
    });

    // Application-specific event handlers
    newSocket.on('connected', (data) => {
      console.log('WebSocket server says:', data.message);
    });

    newSocket.on('joined_simulation', (data) => {
      console.log('Joined simulation updates:', data.message);
    });

    newSocket.on('simulation_update', (data: SocketData) => {
      console.log('Received simulation update:', data);
      setLatestData(data);
    });

    // Reconnection event handlers
    newSocket.on('reconnect', (attemptNumber) => {
      console.log('WebSocket reconnected after', attemptNumber, 'attempts');
      setIsConnected(true);
      setConnectionError(null);
    });

    newSocket.on('reconnect_error', (error) => {
      console.error('WebSocket reconnection error:', error);
      setConnectionError(`Reconnection failed: ${error.message}`);
    });

    setSocket(newSocket);

    // Cleanup on unmount
    return () => {
      console.log('Cleaning up WebSocket connection');
      newSocket.off('connect');
      newSocket.off('disconnect');
      newSocket.off('connect_error');
      newSocket.off('connected');
      newSocket.off('joined_simulation');
      newSocket.off('simulation_update');
      newSocket.off('reconnect');
      newSocket.off('reconnect_error');
      newSocket.close();
    };
  }, []);

  const contextValue: WebSocketContextType = {
    socket,
    isConnected,
    latestData,
    connectionError,
    clearGraphData,
    clearGraphDataTrigger,
  };

  return (
    <WebSocketContext.Provider value={contextValue}>
      {children}
    </WebSocketContext.Provider>
  );
};

export default WebSocketProvider;
