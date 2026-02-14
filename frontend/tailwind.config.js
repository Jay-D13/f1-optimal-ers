/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        f1: {
          red: '#FF1801',
          black: '#15151E',
          white: '#F1F2F3',
          blue: '#0090D0',
        },
        retro: {
          bg: 'var(--retro-bg)',
          text: 'var(--retro-text)',
          border: 'var(--retro-border)',
        }
      },
      fontFamily: {
        mono: ['"Space Mono"', 'monospace'],
        sans: ['Inter', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
