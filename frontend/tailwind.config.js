/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      // Race Programme theme: every colour is a CSS variable from index.css, so light and dark share one set of names
      colors: {
        paper: 'var(--paper)',
        surface: 'var(--surface)',
        ink: 'var(--ink)',
        mute: 'var(--mute)',
        rule: 'var(--rule)',
        accent: 'var(--accent)',
        'on-accent': 'var(--on-accent)',
        model: 'var(--model)',
        pole: 'var(--pole)',
        deploy: 'var(--deploy)',
        harvest: 'var(--harvest)',
        sel: 'var(--sel)',
      },
      fontFamily: {
        display: ['"Big Shoulders Display"', 'sans-serif'],
        num: ['"Courier Prime"', 'monospace'],
        sans: ['"Work Sans"', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
