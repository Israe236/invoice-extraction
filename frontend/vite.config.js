import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The API runs separately (uvicorn on :8000). Proxying /api through the dev
// server keeps the browser on one origin, so CORS never enters the picture
// during development and the frontend needs no environment variable.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
