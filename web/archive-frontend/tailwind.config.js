/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts}"],
  theme: {
    extend: {
      // Pipeline-specific colors matching schema color fields
      colors: {
        pipeline: {
          blue: {
            DEFAULT: "#3b82f6",
            light: "#60a5fa",
            dark: "#2563eb",
          },
          purple: {
            DEFAULT: "#8b5cf6",
            light: "#a78bfa",
            dark: "#7c3aed",
          },
          orange: {
            DEFAULT: "#f97316",
            light: "#fb923c",
            dark: "#ea580c",
          },
          teal: {
            DEFAULT: "#14b8a6",
            light: "#2dd4bf",
            dark: "#0d9488",
          },
          green: {
            DEFAULT: "#22c55e",
            light: "#4ade80",
            dark: "#16a34a",
          },
          pink: {
            DEFAULT: "#ec4899",
            light: "#f472b6",
            dark: "#db2777",
          },
        },
      },
      // Custom animations for UI feedback
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "slide-up": "slideUp 0.3s ease-out",
        "slide-down": "slideDown 0.3s ease-out",
        "fade-in": "fadeIn 0.2s ease-out",
      },
      keyframes: {
        slideUp: {
          "0%": { transform: "translateY(100%)", opacity: "0" },
          "100%": { transform: "translateY(0)", opacity: "1" },
        },
        slideDown: {
          "0%": { transform: "translateY(-100%)", opacity: "0" },
          "100%": { transform: "translateY(0)", opacity: "1" },
        },
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
      },
      // Touch-friendly sizing
      spacing: {
        touch: "44px", // Minimum touch target size
      },
      minWidth: {
        touch: "44px",
      },
      minHeight: {
        touch: "44px",
      },
    },
  },
  plugins: [],
  // Dark mode by default (matches existing UI)
  darkMode: "class",
};
