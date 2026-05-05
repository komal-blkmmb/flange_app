import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  // Set base to your GitHub repo name for GitHub Pages:
  // e.g. if repo is https://github.com/user/flange-app → base: '/flange-app/'
  base: process.env.VITE_BASE_PATH ?? '/',
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  server: {
    port: 5173,
    proxy: {
      // During local dev, proxy API calls to the FastAPI backend
      '/api': {
        target: process.env.VITE_API_URL ?? 'http://localhost:7860',
        changeOrigin: true,
      },
      '/ws': {
        target: process.env.VITE_API_URL ?? 'http://localhost:7860',
        changeOrigin: true,
        ws: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
})
