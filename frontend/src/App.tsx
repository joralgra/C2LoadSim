import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  AppBar,
  Toolbar,
  Alert,
  Snackbar,
} from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { simulatorAPI } from './api/simulatorAPI';
import SimulatorDashboard from './components/SimulatorDashboard';
import { WebSocketProvider } from './contexts/WebSocketProvider';

// Create a dark theme
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#f48fb1',
    },
  },
});

function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  useEffect(() => {
    const checkConnection = async () => {
      try {
        await simulatorAPI.healthCheck();
        setIsConnected(true);
        setConnectionError(null);
      } catch (error) {
        setIsConnected(false);
        setConnectionError('Failed to connect to simulator backend');
        console.error('Connection error:', error);
      }
    };

    // Check connection on mount
    checkConnection();

    // Check connection periodically
    const interval = setInterval(checkConnection, 30000); // Every 30 seconds

    return () => clearInterval(interval);
  }, []);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <WebSocketProvider>
        <Box sx={{ flexGrow: 1 }}>
          <AppBar position="static">
            <Toolbar>
              <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                C2LoadSim - Computation Load Simulator
              </Typography>
              {isConnected ? (
                <Alert severity="success" variant="outlined" sx={{ border: 'none', color: 'white' }}>
                  Connected
                </Alert>
              ) : (
                <Alert severity="error" variant="outlined" sx={{ border: 'none', color: 'white' }}>
                  Disconnected
                </Alert>
              )}
            </Toolbar>
          </AppBar>
          
          <Container maxWidth="xl" sx={{ mt: 3 }}>
            {isConnected ? (
              <SimulatorDashboard />
            ) : (
              <Box sx={{ display: 'flex', justifyContent: 'center', mt: 10 }}>
                <Alert severity="error">
                  Cannot connect to simulator backend. Please ensure the API server is running on port 5000.
                </Alert>
              </Box>
            )}
          </Container>

          <Snackbar 
            open={!!connectionError} 
            autoHideDuration={6000} 
            onClose={() => setConnectionError(null)}
          >
            <Alert onClose={() => setConnectionError(null)} severity="error">
              {connectionError}
            </Alert>
          </Snackbar>
        </Box>
      </WebSocketProvider>
    </ThemeProvider>
  );
}

export default App;
