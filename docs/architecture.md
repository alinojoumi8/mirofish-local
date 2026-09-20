# Architecture

MiroFish-Offline is a local web application with three distinct persistence
responsibilities. Keeping those responsibilities separate prevents the graph
database, workflow state, and large artifacts from becoming coupled.

## Runtime boundaries

```text
Vue routes and components
          |
          v
Flask API blueprints
  graph | simulation | diagnostics | report | market | status
          |
          v
Application services
  graph build/search | simulation lifecycle | forecasting | reporting
       |                     |                         |
       v                     v                         v
Neo4j graph store     SQLite control database     Artifact filesystem
entities, edges,      projects, cases, tasks,     documents, extracted
memories, vectors     simulation run state        text, caches, run files
```

`create_app()` owns startup composition. It initializes Neo4j and embeddings,
opens the SQLite control database, performs the one-time legacy import, wires
repositories into the managers, reconciles persisted simulation processes, and
then registers API blueprints. Runtime dependencies are exposed through
`app.extensions`.

## Backend modules

- `app/api/` contains HTTP translation and response envelopes. Simulation
  diagnostics are separate from lifecycle endpoints.
- `app/services/` owns workflows and provider-independent domain behavior.
  `control_plane.py` is the startup boundary for durable metadata and restart
  reconciliation.
- `app/storage/` owns Neo4j, embeddings, extraction caches, and SQLite. The
  SQLite repository uses WAL mode, a busy timeout, and short transactions.
- `app/models/` defines project, case, and task state and delegates persistence
  to the configured repository.
- `app/utils/` contains provider clients, safe URL fetching, logging, retries,
  file parsing, and market-data adapters.

## Frontend modules

Routes are lazy-loaded from `src/router/routes.js`, so each workflow page is a
separate bundle. Shared helpers own API base-URL selection, escaped Markdown
rendering, and idempotent polling. `SafeMarkdown.vue` is the only component
allowed to use `v-html`; it receives output that has already escaped all raw
HTML.

## Failure and restart behavior

- A missing or malformed legacy JSON record aborts the legacy import as one
  transaction. The source file is left untouched and the import retries after
  the record is repaired and the app restarts.
- Running simulations are reconciled on startup. Persisted processes that no
  longer exist are marked stopped with an interruption reason rather than
  remaining permanently "running."
- SQLite initialization failure is visible in logs and `GET /api/status`; the
  legacy filesystem managers remain available for that process.
- Internal exceptions are logged with a request ID. Default 5xx responses omit
  tracebacks and return the same request ID for correlation.

## Trust boundary

The default backend and every published Compose port bind to loopback. CORS is
restricted to the two local frontend origins. Raw model/user HTML is escaped,
URL fetching validates every redirect target, debug mode is off, and market
network calls are disabled until `MARKET_DATA_PROVIDER=auto` is explicitly set.
