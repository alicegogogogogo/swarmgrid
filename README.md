# SwarmGrid

SwarmGrid is a multi-agent path finding (MAPF) scheduling service. It stores
rectangular grid maps, accepts missions that move several agents at once, plans
each agent in space and time, resolves vertex conflicts and edge swaps by
replanning in priority order, and publishes the per-agent step paths together
with the total makespan.

The initial release intentionally supports a compact public contract:

- a map is a rectangular 4-connected grid plus one obstacle character;
- agents move one cell per timestep (up, down, left, right) or wait in place;
- an agent is **finished** once it reaches its goal; it occupies no cell from
  the next tick onwards, so a finished agent never blocks anyone;
- a **vertex conflict** is two agents occupying one cell at one timestep, which
  includes two agents trying to settle on the same goal cell;
- an **edge swap** is two agents crossing the same edge in opposite directions
  in one timestep;
- following is legal: an agent may move into a cell another agent leaves on the
  same tick, because the two never share the cell;
- planning is prioritized planning: agents are planned in priority order and the
  low-level planner treats the paths of all higher-priority agents as
  space-time obstacles. Each agent is also offered an optional head start of
  waiting on its start cell before it enters the grid, which is what lets the
  lower-priority agent go second around a bottleneck;
- when a priority order fails, the next order is tried, up to a deterministic
  budget; if every order fails, a conflict-tree search re-plans agents under
  explicit space-time constraints for a bounded number of nodes;
- a mission is `unsolvable` when both stages fail. That verdict means "no plan
  was found within these budgets", **not** a proof that no plan exists: the
  branch-and-bound stages are bounded, and any instance whose only solution needs
  a coordination neither stage can express is reported this way. The blocking
  conflicts are returned so the caller can see what stopped it, and the status
  is never faked in the other direction -- a `solved` response always carries a
  plan that satisfies every invariant below;
- a plan request may attach temporary `closures`: cells that are impassable only
  during a given inclusive tick interval. The planner waits or detours around
  them exactly like moving walls, and an agent sealed on its start at tick zero
  gets a `closure` diagnostic instead of a plan;
- repeating a mutating request with the same idempotency key returns the first
  response.

## Requirements

- Python 3.11 or newer
- no third-party runtime dependencies

## Run the service

```bash
PYTHONPATH=src python -m swarmgrid.server --host 127.0.0.1 --port 8080 --database swarmgrid.db
```

The process prints `SwarmGrid listening on http://127.0.0.1:8080` after it has
bound the port. The database file is created when missing, so state survives a
restart.

## HTTP API

All request and response bodies are JSON. Unknown fields are rejected with
`validation_error`. Every `POST` that stores or changes state requires an
`Idempotency-Key` header. Response objects are serialized with sorted keys, so a
repeated request is byte-identical to the first response; the examples below
show logical field order, not wire order.

### Health

```http
GET /health
```

Returns `{"status":"ok"}`.

### Create a map

```http
POST /maps
Idempotency-Key: warehouse-map-1

{
  "id": "warehouse",
  "width": 5,
  "height": 3,
  "grid": [".....", "..##.", "....."]
}
```

Returns HTTP 201 with the stored map. A map contains exactly `id`, `width`,
`height`, `grid`, and the optional `obstacle` character (default `#`):

- `width` and `height` are integers between 1 and 64; the grid may not exceed
  4096 cells;
- `grid` is an array of exactly `height` strings, each exactly `width`
  characters long, containing only `.` and the obstacle character;
- `obstacle` is a single non-whitespace character.

Coordinates are `[x, y]` with `x` the column and `y` the row, both zero-based.
`(0, 0)` is the top-left cell. A replayed `POST /maps` with the same body and
idempotency key returns HTTP 201 and the original document; a different
idempotency key for the same `id` returns `conflict`.

```http
GET /maps/warehouse
```

Returns the stored map.

### Create a mission

```http
POST /missions
Idempotency-Key: shift-1-request

{
  "id": "shift-1",
  "map_id": "warehouse",
  "agents": [
    {"id": "agv-1", "start": [0, 0], "goal": [4, 0]},
    {"id": "agv-2", "start": [4, 0], "goal": [0, 0]}
  ]
}
```

Returns HTTP 201 with the mission in `created` state:

```json
{
  "id": "shift-1",
  "map_id": "warehouse",
  "status": "created",
  "created_at": "2026-01-01T00:00:00.000000Z",
  "agents": [
    {"id": "agv-1", "start": [0, 0], "goal": [4, 0]},
    {"id": "agv-2", "start": [4, 0], "goal": [0, 0]}
  ]
}
```

A mission contains exactly `id`, `map_id`, and `agents`. Each agent contains
exactly `id`, `start`, and `goal`. Validation rules:

- `map_id` must exist (otherwise `not_found`);
- agent ids must be unique inside the mission;
- start cells must be pairwise distinct and every `start`/`goal` must be inside
  the map and not an obstacle;
- an agent whose start equals its goal is rejected;
- at most 32 agents per mission.

### Plan a mission

```http
POST /missions/shift-1/plan
Idempotency-Key: shift-1-plan
Content-Type: application/json

{}
```

The body is optional; an empty object is enough. The body accepts four optional
fields:

- `priority` — an explicit priority order: every agent id of the mission,
  exactly once. When present the planner uses that order only.
- `order_budget` — how many priority orders the planner may try, an integer
  between 1 and 5040 (default 24, capped by the factorial of the agent count).
- `tree_budget` — how many nodes the conflict-tree stage may expand, an integer
  between 0 and 4096 (default 64). Raise it to search harder before giving up.
- `closures` — task-level temporary road blocks, an array of up to 64 entries.
  Each entry contains exactly `cells`, `start_tick`, and `end_tick`:
  - `cells` is a non-empty array of up to 64 unique `[x, y]` points, each inside
    the map and not a static obstacle;
  - `start_tick` and `end_tick` are integers with
    `0 <= start_tick <= end_tick <= 100000`.

  The tick interval is inclusive on both ends. During every timestep
  `start_tick..end_tick` none of the listed cells may be entered, waited on, or
  held by any agent; outside the interval the cells are ordinary ground. The
  closures feed both the space-time search and conflict resolution, so a plan is
  delayed or detoured around a closed cell rather than touching it. Omitting
  `closures` (or sending `[]`) means no temporary closure, and the request is
  then identical to the body-less case.

A solved response (HTTP 200):

```json
{
  "mission_id": "shift-1",
  "map_id": "warehouse",
  "status": "solved",
  "method": "priority",
  "makespan": 8,
  "priority_order": ["agv-1", "agv-2"],
  "orders_tried": 1,
  "orders_budget": 2,
  "agents": [
    {
      "agent_id": "agv-1",
      "start": [0, 0],
      "goal": [4, 0],
      "arrival": 4,
      "path": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]
    },
    {
      "agent_id": "agv-2",
      "start": [4, 0],
      "goal": [0, 0],
      "arrival": 8,
      "path": [[4, 0], [4, 1], [4, 2], [3, 2], [2, 2], [1, 2], [0, 2],
               [0, 1], [0, 0]]
    }
  ]
}
```

`agents` is sorted by `agent_id`. `path[0]` is the agent's `start`, `path[-1]`
its `goal`, and `path[t]` is the cell occupied at timestep `t`. Consecutive
cells in a path are identical (a wait) or 4-adjacent (a move). `arrival` is
`len(path) - 1`, the timestep at which the agent reaches its goal; the agent is
finished from then on. `makespan` is the largest `arrival` over all agents, so
the schedule spans timesteps `0..makespan`. `method` is `priority` when a
priority order produced the plan and `constraint_tree` when the fallback search
did.

An unsolvable response (HTTP 200), for the mission `swap` on the one-row map
`{"id":"lane","width":4,"height":1,"grid":["...."]}` with two agents that must
trade ends:

```json
{
  "mission_id": "swap",
  "map_id": "lane",
  "status": "unsolvable",
  "method": null,
  "makespan": null,
  "priority_order": [],
  "orders_tried": 2,
  "orders_budget": 2,
  "conflicts": [
    {
      "type": "vertex",
      "timestep": 3,
      "agent": "a",
      "other": "b",
      "position": [0, 0],
      "other_position": [3, 0],
      "message": "agent a is trapped on [0, 0]: b finishes there at t=3"
    },
    {
      "type": "vertex",
      "timestep": 3,
      "agent": "b",
      "other": "a",
      "position": [3, 0],
      "other_position": [0, 0],
      "message": "agent b is trapped on [3, 0]: a finishes there at t=3"
    }
  ]
}
```

`unsolvable` means no collision-free joint plan was found inside the budgets, so
no path data is returned. `conflicts` is sorted and de-duplicated, and each
entry has these fields:

- `type` is `vertex`, `edge`, `unreachable_goal`, or `closure`;
- `timestep` is the timestep at which the collision occurs, or the tick that
  blocks the agent; `0` for an `unreachable_goal`;
- `agent` is the agent that could not be routed;
- `other` is the other agent involved, or `null`;
- `position` and `other_position` are the cells involved, or `null`;
- `message` is a human readable sentence.

A `closure` entry names a temporary road block that stops an agent: `timestep`
is the first forbidden tick the search needs, `position` is the closed cell,
`other` and `other_position` are `null`, and the message states that the closure
blocks that agent. When an agent's start is covered by a closure whose
`start_tick` is `0`, the agent cannot even occupy its first cell, so the
mission is reported `unsolvable` without trying orders: one diagnostic per
blocked agent is returned, sorted by `agent_id`, each with `timestep` `0` and
`position` equal to that agent's start.

A malformed `closures` value (bad shape, out-of-range ticks, an out-of-bounds or
obstacle cell, too many entries or cells, and so on) is a `validation_error`.

Plans are deterministic: the same map, mission, and request body always return
the same plan. Only solved plans are stored; plan an `unsolvable` mission again
to retry it (for example with a larger `order_budget` or `tree_budget`).
Repeated planning of a solved mission returns the recorded plan and does not
replan.

### Inspect a mission

```http
GET /missions/shift-1
```

Returns the mission state, including `planned_at` once it has been planned, and
`plan` — the exact object described above, or `null` when no solved plan has
been recorded.

### Read the schedule

```http
GET /missions/shift-1/schedule
```

Returns HTTP 200 with one frame per timestep:

```json
{
  "mission_id": "shift-1",
  "map_id": "warehouse",
  "makespan": 8,
  "priority_order": ["agv-1", "agv-2"],
  "timeline": [
    {"timestep": 0, "positions": {"agv-1": [0, 0], "agv-2": [4, 0]}},
    {"timestep": 1, "positions": {"agv-1": [1, 0], "agv-2": [4, 1]}}
  ]
}
```

`timeline` has exactly `makespan + 1` frames. A frame maps every agent to its
cell, or to `null` when the agent has already finished, so a finished agent
never blocks a cell. No frame puts two agents on one cell and no frame contains
an edge swap, so the schedule is directly renderable.

## Invariants

Every published plan satisfies all of the following; the service re-checks them
before responding and returns `conflict` rather than serving a broken plan:

1. each path is continuous (consecutive cells are identical or 4-adjacent),
   starts on the agent's start cell, and ends on its goal cell;
2. no path ever enters an obstacle or leaves the map, and no path occupies a cell
   on a tick for which a temporary `closures` entry closes that cell;
3. no two agents occupy the same cell at the same timestep, and two agents can
   never settle on one cell;
4. no two agents trade cells along an edge in one timestep;
5. `makespan` equals the largest `arrival`, and `arrival` is the last timestep
   the agent occupies a cell;
6. identical input always yields an identical plan.

## Errors

Errors use this shape:

```json
{"error":{"code":"validation_error","message":"human readable detail"}}
```

Validation errors return 400, missing resources return 404, and conflicts
return 409. The codes are `validation_error`, `not_found`, `conflict`, and
`internal_error`. An unsatisfiable mission is **not** an error response:
`POST /missions/{id}/plan` answers HTTP 200 with `"status":"unsolvable"`.
`GET /missions/{id}/schedule` returns `conflict` when the mission has no
recorded solved plan, and `GET /maps/{id}` returns `not_found` for an unknown
map.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

`tests/test_service.py` covers map and mission validation, idempotency, the
planner invariants, unsolvable missions, and persistence across a restart.
`tests/test_server.py` starts the real HTTP server on an ephemeral port and
covers routing, status codes, and the error envelope.

Two extra scripts are provided for manual checks. Neither is needed at runtime:

```bash
bash scripts/smoke.sh            # start on 127.0.0.1:18080, curl every route, kill it
PYTHONPATH=src python scripts/verify_solver.py   # randomized + brute-force planner audit
```

`smoke.sh` prints the served responses and stops the process it started.
`verify_solver.py` plans hundreds of randomly generated maps and re-checks every
published plan, then compares `unsolvable` verdicts on small maps against an
exhaustive joint-path search.
