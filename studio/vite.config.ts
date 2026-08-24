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
    // The output directory is tracked (a lone .gitkeep) because api.py mounts
    // StaticFiles on it at import, and emptying it would delete that marker on
    // every build. The bundle itself is gitignored, and the Docker builder
    // stage starts from a clean tree, so nothing stale can ship.
    emptyOutDir: false,
  },
  test: {
    environment: "jsdom",
    setupFiles: "./src/test-setup.ts",
    exclude: [...configDefaults.exclude, "e2e/**"],
  },
});
