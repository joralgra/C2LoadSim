import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';

// Simple test to verify testing setup works
test('renders basic component', () => {
  const TestComponent = () => <div>Hello World</div>;
  render(<TestComponent />);
  expect(screen.getByText('Hello World')).toBeInTheDocument();
});
