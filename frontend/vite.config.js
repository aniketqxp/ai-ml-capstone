import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const staticArtifacts = process.env.VITE_STATIC_ARTIFACTS === '1'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: staticArtifacts ? undefined : {
      '/calls': 'http://127.0.0.1:8000',
      '/sentence_segments': 'http://127.0.0.1:8000',
      '/sentiment': 'http://127.0.0.1:8000',
      '/audio': 'http://127.0.0.1:8000',
    },
  },
})
