import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/search': 'http://127.0.0.1:8000',
      '/filter': 'http://127.0.0.1:8000',
      '/compare': 'http://127.0.0.1:8000',
    }
  }
})
