/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Pipeline accent colors
        pipeline: {
          blue: '#3b82f6',    // Z-Image
          purple: '#a855f7',  // LTX-2
          orange: '#f97316',  // FLUX.2
          teal: '#14b8a6',    // Qwen T2I
          green: '#22c55e',   // Qwen Edit
          pink: '#ec4899',    // Qwen Layered
        },
      },
      minHeight: {
        touch: '44px',
      },
      minWidth: {
        touch: '44px',
      },
      spacing: {
        // Based on 8px grid
        18: '4.5rem',  // 72px
        22: '5.5rem',  // 88px
      },
    },
  },
  plugins: [],
  // Dark mode only - no theme switching
  darkMode: 'class',
}
