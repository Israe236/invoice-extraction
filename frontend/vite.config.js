import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The API runs separately (uvicorn on :8000). Proxying /api through the dev
// server keeps the browser on one origin, so CORS never enters the picture
// during development and the frontend needs no environment variable.
// Ports are overridable because 8000 and 5173 are the defaults for half the
// dev tools on a machine. If another app already holds 8000, the proxy would
// silently forward uploads to it instead of to this API.
const apiUrl = process.env.API_URL ?? 'http://127.0.0.1:8000'
const port = Number(process.env.PORT ?? 5173)

export default defineConfig({
  plugins: [react()],
  server: {
    port,
    strictPort: true,
    proxy: {
      '/api': {
        target: apiUrl,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
