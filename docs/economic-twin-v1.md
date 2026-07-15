# Economic Twin v1

Economic Twin adds an opt-in, settled economy to the existing OASIS social
simulation. Twitter and Reddit still provide the agent interaction layer, while
one shared SQLite kernel owns identity, money, jobs, trades, events, and metrics.

## Enable it

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
- Provider or JSON failures reject affected intents without stopping the social
  simulation. A ledger invariant failure halts only the economic runtime.
- Settled balances, jobs, and events are injected into each agent before its
  next social action.

Supported intents are `post_job`, `apply_job`, `hire`, `complete_job`,
`transfer`, `buy_goods`, and `do_nothing`.

Each simulation stores its state in `backend/app/uploads/simulations/<id>/economy.db`.
Force restart removes the database and its SQLite sidecar files.

## Read APIs

- `GET /api/simulation/<id>/economy/summary`
- `GET /api/simulation/<id>/economy/agents`
- `GET /api/simulation/<id>/economy/jobs`
- `GET /api/simulation/<id>/economy/events`
- `GET /api/simulation/<id>/economy/ledger`

List endpoints accept `limit` and `offset`. The report agent also has an
`economy_evidence` tool that returns settled metrics, identities, jobs, trades,
events, and ledger transactions.

## Scope boundary

Version one deliberately excludes banking, credit, equity, taxation,
government, and demographic lifecycle mechanics. Those can be layered on the
ledger later without making social posts authoritative economic state.
