# SQLite control-plane migration

The control database stores small workflow records that need transactional
updates: projects, cases, tasks, and simulation run state. Neo4j remains the
knowledge-graph store, while documents, extracted text, caches, and other large
artifacts remain under `backend/uploads/`.

## First startup

The default database is `backend/uploads/mirofish.db`. On first startup the app
scans the existing `project.json`, `case.json`, and `run_state.json` records and
imports them in one SQLite transaction. Existing database rows win, so a retry
never overwrites newer durable state. After a successful import, a migration
marker prevents repeated scans.

Legacy JSON files are not deleted or rewritten by the import. Simulation
`run_state.json` files continue to be updated as compatibility mirrors for the
simulation subprocess IPC boundary. SQLite is authoritative for application
metadata after migration.

## Back up safely

Stop the application before copying the database so its WAL is fully
checkpointed, then copy the database and artifacts together:

```bash
cp backend/uploads/mirofish.db /path/to/backup/mirofish.db
cp -a backend/uploads/projects /path/to/backup/
cp -a backend/uploads/cases /path/to/backup/
cp -a backend/uploads/simulations /path/to/backup/
```

For Docker, stop the `mirofish` service first and back up the bind-mounted
`backend/uploads` directory. Neo4j data remains in its separate named volume and
needs its own backup strategy.

## Observe the migration

`GET /api/status` exposes:

- `control_db.healthy`, path, schema version, and journal mode;
- `control_db.migration`, including imported record counts;
- `control_db.runner_reconciliation`, including interrupted stale runs.

Application logs also record the database path and import counts at startup.

## Recover from a failed import

A malformed or unreadable legacy record prevents the migration marker from
being written. The app logs the exact file, leaves every source record intact,
and stops startup to prevent reads or writes against a stale filesystem snapshot.
Repair or restore the named JSON file and restart; the complete import will retry.

Do not delete legacy files immediately after migrating. They are a migration
snapshot, but they do not receive later project/case/task updates and therefore
are not a current rollback source. Back up `mirofish.db` regularly once the app
has accepted new work.
