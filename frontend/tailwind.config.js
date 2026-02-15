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
        },
        panel: {
          bg: 'var(--panel-bg)',
          muted: 'var(--panel-muted)',
          border: 'var(--panel-border)',
        }
      },
      fontFamily: {
        mono: ['"Space Mono"', 'monospace'],
        sans: ['"IBM Plex Sans"', 'Inter', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
