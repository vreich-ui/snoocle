import { configDefaults, defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/studio/",
  plugins: [
    react(),
    {
      name: "strip-generated-trailing-whitespace",
      renderChunk(code) {
        return code.replace(/[ \t]+$/gm, "");
      },
    },
  ],
  build: {
    outDir: "../snoocle_server/studio",
    emptyOutDir: true,
  },
  test: {
    environment: "jsdom",
    setupFiles: "./src/test-setup.ts",
    exclude: [...configDefaults.exclude, "e2e/**"],
  },
});
