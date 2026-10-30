/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string
  readonly VITE_APP_TITLE: string
  readonly VITE_VISUAL_SNAPSHOTS_PATH: string
  readonly VITE_NEO4J_URL: string
  readonly VITE_NEO4J_DATABASE: string
  readonly VITE_NEO4J_USER: string
  readonly VITE_NEO4J_PASSWORD: string
  readonly VITE_NEO4J_GRAPH_LIMIT: string
  readonly VITE_NEO4J_GRAPH_SEED_LIMIT: string
  readonly VITE_NEO4J_GRAPH_QUERY: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
