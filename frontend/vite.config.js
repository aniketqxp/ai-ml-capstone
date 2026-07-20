import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/calls': 'http://127.0.0.1:8000',
      '/sentence_segments': 'http://127.0.0.1:8000',
      '/sentiment': 'http://127.0.0.1:8000',
      '/audio': 'http://127.0.0.1:8000',
    },
  },
})
