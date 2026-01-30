import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
  root: ".",
  base: "/",

  // Dev server configuration
  server: {
    port: 5173,
    strictPort: true,
    // Proxy API requests to FastAPI backend during development
    proxy: {
      "/api": {
        target: "http://localhost:7860",
        changeOrigin: true,
      },
      // Proxy generated images/videos
      "/outputs": {
        target: "http://localhost:7860",
        changeOrigin: true,
      },
    },
  },

  // Build configuration
  build: {
    outDir: "dist",
    emptyOutDir: true,
    // Generate source maps for debugging
    sourcemap: true,
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
      },
    },
  },

  // Resolve TypeScript paths
  resolve: {
    alias: {
      "@": resolve(__dirname, "src"),
      "@components": resolve(__dirname, "src/components"),
      "@core": resolve(__dirname, "src/core"),
      "@types": resolve(__dirname, "src/types"),
      "@pipelines": resolve(__dirname, "src/pipelines"),
    },
  },
});
