import { createServer } from 'vite';

process.env.VITE_EVALUATOR_V2 = '1';
process.env.VITE_STATIC_ARTIFACTS = '1';

const server = await createServer({
  server: {
    host: '127.0.0.1',
    port: 5173,
  },
});

await server.listen();
server.printUrls();
