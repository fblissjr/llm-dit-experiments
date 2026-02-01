/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        pipeline: {
          blue: '#3b82f6',
          purple: '#a855f7',
          orange: '#f97316',
          teal: '#14b8a6',
          green: '#22c55e',
          pink: '#ec4899',
        },
      },
      minHeight: {
        touch: '44px',
      },
      minWidth: {
        touch: '44px',
      },
    },
  },
  plugins: [],
  darkMode: 'class',
}
