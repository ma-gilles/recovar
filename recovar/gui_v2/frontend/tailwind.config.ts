import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Design system tokens
        app: "var(--bg-app, #09090b)",
        surface: "var(--bg-surface, #18181b)",
        "surface-hover": "var(--bg-surface-hover, #27272a)",
        "surface-active": "var(--bg-surface-active, #3f3f46)",
        input: "var(--bg-input, #27272a)",
        accent: {
          DEFAULT: "var(--accent, #2563eb)",
          hover: "var(--accent-hover, #3b82f6)",
        },
      },
    },
  },
  plugins: [],
};
export default config;
