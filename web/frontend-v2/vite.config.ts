import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5175,
    host: true,
    proxy: {
      '/api': {
        target: 'http://localhost:7860',
        changeOrigin: true,
        // Required for SSE streaming - don't buffer the response
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            // Check if this is an SSE response
            if (proxyRes.headers['content-type']?.includes('text/event-stream')) {
              // Disable buffering for streaming responses
              proxyRes.headers['cache-control'] = 'no-cache';
              proxyRes.headers['connection'] = 'keep-alive';
            }
          });
        },
      },
      '/outputs': {
        target: 'http://localhost:7860',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
})
