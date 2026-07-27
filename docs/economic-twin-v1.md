# Economic Twin v1

Economic Twin adds an opt-in, settled economy to the existing OASIS social
simulation. Twitter and Reddit still provide the agent interaction layer, while
one shared SQLite kernel owns identity, money, jobs, trades, events, and metrics.

## Enable it

In the application, finish Step 2 and turn on **Economic Twin** before starting
the simulation. Configure the initial balance in cents, a 3-8 character
currency code, and 1-500 agent decisions per tick. The control is off by
default, and Step 3 preserves that choice instead of enabling the economy
implicitly.

Pass an `economy` object to `POST /api/simulation/start`:

```json
{
  "simulation_id": "sim_example",
  "platform": "parallel",
  "economy": {
    "enabled": true,
    "initial_balance_cents": 1000000,
    "max_decisions_per_tick": 50,
    "currency": "USD"
  }
}
```

The server defaults to `enabled: false`. Unless `rounds_per_tick` is supplied,
one economic tick represents one simulated day:
`ceil(1440 / minutes_per_round)`.

Omitting `economy` remains backward compatible and does not create an economic
database.

## Runtime contract

- One economic identity is derived from each source entity UUID and reused by
  both social platforms.
- All monetary values are integer cents.
- Transfers and trades reject missing counterparties, self-payment,
  insufficient funds, and non-positive amounts.
- Job transitions are `open -> hired -> completed`; applications and employer
  authority are validated before a transition.
- Money moves through two-entry transactions whose entries sum to zero.
- Every intent has a unique idempotency key, and every economic tick has a
  persisted exact-once claim.
- At an economic boundary, `EconomyTickCoordinator` waits until every selected
  social platform has persisted the current round. In a parallel run the first
  of Twitter or Reddit waits; the second arrival executes settlement once and
  releases both loops. Twitter-only and Reddit-only runs settle immediately.
- Provider or JSON failures reject affected intents without stopping the social
  simulation. Tick results include one rejection result per selected agent so
  provider failures are visible to callers. A ledger invariant failure halts
  only the economic runtime.
- A process restart marks any leftover `running` tick as `failed`, records a
  `tick_interrupted` event, and continues with later ticks. Interrupted ticks
  are never replayed, preserving exact-once money movement.

Supported intents are `post_job`, `apply_job`, `hire`, `complete_job`,
`transfer`, `buy_goods`, and `do_nothing`.

Each simulation stores its state in `backend/app/uploads/simulations/<id>/economy.db`.
Force restart removes the database and its SQLite sidecar files.

Every database operation owns and explicitly closes its SQLite connection.
APIs and report generation open `EconomyStore(..., readonly=True)`, which never
creates directories, initializes schema, records metrics, or changes the
database. This also allows the database to be deleted immediately after a run
on Windows.

## Provider behavior

Economic decisions call `LLMClient.chat_json(..., disable_reasoning=True)`.
For Ollama this uses the native `/api/chat` endpoint with `stream: false`,
`format: "json"`, and `think: false`; this is required for reasoning models such
as `qwen3.5:4b`, whose OpenAI-compatible response may contain reasoning but an
empty content field. Cloud and other non-Ollama transports retain their existing
OpenAI-compatible behavior.

Malformed or empty provider output is recorded as rejected intents. It never
falls back to guessed economic actions and never stops the social simulation.

## Read APIs

- `GET /api/simulation/<id>/economy/summary`
- `GET /api/simulation/<id>/economy/agents`
- `GET /api/simulation/<id>/economy/jobs`
- `GET /api/simulation/<id>/economy/events`
- `GET /api/simulation/<id>/economy/ledger`

The summary includes total, completed, running, and failed tick counts plus the
latest tick status/error. List endpoints accept `limit` and `offset`.

When the economy is enabled, report generation automatically prefetches one
bounded settled snapshot for the outline and every section prompt, and records
the source as `economy_evidence` in section evidence cards. The tool remains
available for deeper queries. If the economic database is missing, the report
receives an explicit evidence warning and is instructed not to fabricate
balances, jobs, trades, prices, or economic outcomes.

## Release and smoke tests

Run the deterministic gates from the repository root:

```powershell
uv run --project backend python -m pytest backend/tests -q
npm --prefix frontend test
npm --prefix frontend run build
git diff --check
```

With Ollama running and `qwen3.5:4b` installed, run the live gate:

```powershell
uv run --project backend python backend/scripts/smoke_economy.py --model qwen3.5:4b
```

The command requires one completed tick, one result per selected agent, no
provider-format rejection, a balanced ledger, and successful removal of the
temporary SQLite database directory. Override `--base-url` or `--api-key` when
the local Ollama endpoint differs from `http://localhost:11434/v1`.

## Scope boundary

Version one deliberately excludes firms, production, inventory, dynamic
pricing, contracts, credit, taxation, government, banking, equity, and
demographic lifecycle mechanics. Those are v2 work and can be layered on the
ledger later without making social posts authoritative economic state.
