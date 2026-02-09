import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";
import fs from "fs";

const backendUrl = process.env.VITE_BACKEND_URL || "https://localhost:7860";
const sslCert = process.env.VITE_SSL_CERT;
const sslKey = process.env.VITE_SSL_KEY;

export default defineConfig({
    plugins: [react()],
    resolve: {
        alias: {
            "@": resolve(__dirname, "./src"),
        },
    },
    server: {
        port: 5175,
        host: true,
        https:
            sslCert && sslKey
                ? {
                      cert: fs.readFileSync(sslCert),
                      key: fs.readFileSync(sslKey),
                  }
                : undefined,
        proxy: {
            "/api": {
                target: backendUrl,
                changeOrigin: true,
                secure: false,
                // Required for SSE streaming - don't buffer the response
                configure: (proxy) => {
                    proxy.on("proxyRes", (proxyRes) => {
                        // Check if this is an SSE response
                        if (
                            proxyRes.headers["content-type"]?.includes(
                                "text/event-stream",
                            )
                        ) {
                            // Disable buffering for streaming responses
                            proxyRes.headers["cache-control"] = "no-cache";
                            proxyRes.headers["connection"] = "keep-alive";
                        }
                    });
                },
            },
            "/outputs": {
                target: backendUrl,
                changeOrigin: true,
                secure: false,
            },
        },
    },
    build: {
        outDir: "dist",
        sourcemap: true,
    },
});
