import { defineConfig } from 'vite';

export default defineConfig({
  optimizeDeps: {
    exclude: ['opencascade.js']
  },
  build: {
    target: 'esnext'
  },
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp'
    }
  }
});
