/// <reference types="vitest/config" />
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { resolve } from "path";
import fs from "fs";
import { homedir } from "os";

/** Expand leading `~/` to the user's home directory. */
function expandHome(p: string): string {
    return p.startsWith("~/") ? resolve(homedir(), p.slice(2)) : p;
}

export default defineConfig(({ mode }) => {
    // loadEnv with empty prefix loads ALL env vars, not just VITE_-prefixed ones.
    // The object form of defineConfig evaluates before Vite loads .env files,
    // so process.env.VITE_* would always be empty there.
    const env = loadEnv(mode, process.cwd(), "");

    const backendUrl = env.VITE_BACKEND_URL || "https://localhost:7860";
    const sslCert = env.VITE_SSL_CERT;
    const sslKey = env.VITE_SSL_KEY;

    let httpsConfig: { cert: Buffer; key: Buffer } | undefined;
    if (sslCert && sslKey) {
        try {
            httpsConfig = {
                cert: fs.readFileSync(expandHome(sslCert)),
                key: fs.readFileSync(expandHome(sslKey)),
            };
        } catch (e) {
            console.warn("SSL cert/key not found, falling back to HTTP:", e);
        }
    }

    return {
        plugins: [tailwindcss(), react()],
        resolve: {
            alias: {
                "@": resolve(__dirname, "./src"),
            },
        },
        server: {
            port: 5175,
            host: true,
            https: httpsConfig,
            proxy: {
                "/api": {
                    target: backendUrl,
                    changeOrigin: true,
                    secure: false,
                    // Required for SSE streaming - don't buffer the response
                    configure: (proxy, _options) => {
                        (proxy as any).on("proxyRes", (proxyRes: any) => {
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
            sourcemap: mode !== 'production',
        },
        test: {
            environment: "jsdom",
            setupFiles: ["./src/__tests__/setup.ts"],
        },
    };
});
