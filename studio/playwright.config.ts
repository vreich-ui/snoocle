import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  expect: { timeout: 10_000 },
  use: {
    baseURL: "http://127.0.0.1:4173",
    trace: "retain-on-failure",
  },
  webServer: {
    command: "npm run build && cd .. && python3 -m uvicorn snoocle_server.api:app --host 127.0.0.1 --port 4173",
    url: "http://127.0.0.1:4173/healthz",
    reuseExistingServer: true,
    timeout: 120_000,
    env: {
      SNOOCLE_STORE: "memory",
      SNOOCLE_API_TOKEN: "browser-test-token",
    },
  },
});
