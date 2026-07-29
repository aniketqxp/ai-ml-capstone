import fs from 'node:fs';

import { defineConfig } from '@playwright/test';

const localChrome =
  process.env.PLAYWRIGHT_CHROME_PATH ||
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 30_000,
  webServer: {
    command: 'npm run dev:e2e',
    url: 'http://127.0.0.1:5173',
    reuseExistingServer: true,
    timeout: 30_000,
  },
  use: {
    baseURL: 'http://127.0.0.1:5173',
    browserName: 'chromium',
    launchOptions: fs.existsSync(localChrome)
      ? { executablePath: localChrome }
      : {},
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
  },
});
