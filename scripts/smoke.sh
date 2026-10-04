#!/usr/bin/env bash
# Smoke test for the SwarmGrid service: start it, hit every route, then kill it.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

rm -f /tmp/sg.db /tmp/sg.db-wal /tmp/sg.db-shm /tmp/sg.log
PYTHONPATH=src nohup python3 -m swarmgrid.server --host 127.0.0.1 --port 18080 --database /tmp/sg.db >/tmp/sg.log 2>&1 &
SERVER_PID=$!
sleep 2

BASE=http://127.0.0.1:18080
JSON='Content-Type: application/json'

echo "--- server stdout ---"
cat /tmp/sg.log
echo "--- GET /health ---"
curl -s "$BASE/health"; echo
echo "--- POST /maps ---"
curl -s -X POST "$BASE/maps" -H "$JSON" -H 'Idempotency-Key: warehouse-map-1' \
  -d '{"id":"warehouse","width":5,"height":3,"grid":[".....","..##.","....."]}'; echo
echo "--- POST /maps (replayed key) ---"
curl -s -X POST "$BASE/maps" -H "$JSON" -H 'Idempotency-Key: warehouse-map-1' \
  -d '{"id":"warehouse","width":5,"height":3,"grid":[".....","..##.","....."]}'; echo
echo "--- POST /missions ---"
curl -s -X POST "$BASE/missions" -H "$JSON" -H 'Idempotency-Key: shift-1-request' \
  -d '{"id":"shift-1","map_id":"warehouse","agents":[{"id":"agv-1","start":[0,0],"goal":[4,0]},{"id":"agv-2","start":[4,0],"goal":[0,0]}]}'; echo
echo "--- POST /missions/shift-1/plan ---"
curl -s -X POST "$BASE/missions/shift-1/plan" -H "$JSON" -H 'Idempotency-Key: shift-1-plan' -d '{}'; echo
echo "--- GET /missions/shift-1 ---"
curl -s "$BASE/missions/shift-1"; echo
echo "--- GET /missions/shift-1/schedule ---"
curl -s "$BASE/missions/shift-1/schedule"; echo
echo "--- POST /missions/shift-1/replan ---"
curl -s -X POST "$BASE/missions/shift-1/replan" -H "$JSON" -H 'Idempotency-Key: shift-1-replan-1' \
  -d '{"observed_tick":2,"positions":{"agv-1":[2,0],"agv-2":[4,2]}}'; echo
echo "--- POST /missions/shift-1/replan (replayed key) ---"
curl -s -X POST "$BASE/missions/shift-1/replan" -H "$JSON" -H 'Idempotency-Key: shift-1-replan-1' \
  -d '{"observed_tick":2,"positions":{"agv-1":[2,0],"agv-2":[4,2]}}'; echo
echo "--- temporary closure: POST /missions/closed/plan ---"
curl -s -X POST "$BASE/missions" -H "$JSON" -H 'Idempotency-Key: closed-request' \
  -d '{"id":"closed","map_id":"warehouse","agents":[{"id":"agv-1","start":[0,0],"goal":[4,0]}]}' >/dev/null
curl -s -X POST "$BASE/missions/closed/plan" -H "$JSON" -H 'Idempotency-Key: closed-plan' \
  -d '{"closures":[{"cells":[[1,0]],"start_tick":1,"end_tick":1}]}'; echo
echo "--- closure blocks the start at t=0: POST /missions/sealed/plan ---"
curl -s -X POST "$BASE/missions" -H "$JSON" -H 'Idempotency-Key: sealed-request' \
  -d '{"id":"sealed","map_id":"warehouse","agents":[{"id":"agv-1","start":[0,0],"goal":[4,0]}]}' >/dev/null
curl -s -X POST "$BASE/missions/sealed/plan" -H "$JSON" -H 'Idempotency-Key: sealed-plan' \
  -d '{"closures":[{"cells":[[0,0]],"start_tick":0,"end_tick":100000}]}'; echo
echo "--- unsolvable: POST /missions/swap/plan ---"
curl -s -X POST "$BASE/maps" -H "$JSON" -H 'Idempotency-Key: lane-map' \
  -d '{"id":"lane","width":4,"height":1,"grid":["...."]}' >/dev/null
curl -s -X POST "$BASE/missions" -H "$JSON" -H 'Idempotency-Key: swap-request' \
  -d '{"id":"swap","map_id":"lane","agents":[{"id":"a","start":[0,0],"goal":[3,0]},{"id":"b","start":[3,0],"goal":[0,0]}]}' >/dev/null
curl -s -X POST "$BASE/missions/swap/plan" -H "$JSON" -H 'Idempotency-Key: swap-plan' -d '{}'; echo
echo "--- unsolvable: GET /missions/swap/schedule ---"
curl -s -o /dev/null -w 'HTTP %{http_code} ' "$BASE/missions/swap/schedule"
curl -s "$BASE/missions/swap/schedule"; echo
echo "--- error: unknown field ---"
curl -s -o /dev/null -w 'HTTP %{http_code} ' -X POST "$BASE/maps" -H "$JSON" -H 'Idempotency-Key: bad' \
  -d '{"id":"x","width":1,"height":1,"grid":["."],"nope":1}'
curl -s -X POST "$BASE/maps" -H "$JSON" -H 'Idempotency-Key: bad2' \
  -d '{"id":"x","width":1,"height":1,"grid":["."],"nope":1}'; echo
echo "--- error: missing mission ---"
curl -s -o /dev/null -w 'HTTP %{http_code} ' "$BASE/missions/ghost"
curl -s "$BASE/missions/ghost"; echo
echo "--- error: route not found ---"
curl -s -o /dev/null -w 'HTTP %{http_code} ' "$BASE/nowhere"; echo

kill "$SERVER_PID" 2>/dev/null
sleep 1
if kill -0 "$SERVER_PID" 2>/dev/null; then echo "server still alive, forcing"; kill -9 "$SERVER_PID"; fi
echo "--- process check ---"
pgrep -af 'swarmgrid.server' || echo 'no swarmgrid process running'
