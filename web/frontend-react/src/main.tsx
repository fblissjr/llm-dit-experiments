// Enable Immer plugins BEFORE any stores are created
// This must be at the very top - stores are created at module load time
import { enableMapSet } from 'immer';
enableMapSet();

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './styles/main.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
