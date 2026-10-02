import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from swarmgrid.server import Handler
from swarmgrid.service import SwarmGrid

MAP = {"id": "grid", "width": 5, "height": 3, "grid": [".....", "..##.", "....."]}
MISSION = {
    "id": "run-1",
    "map_id": "grid",
    "agents": [
        {"id": "a", "start": [0, 0], "goal": [4, 0]},
        {"id": "b", "start": [4, 0], "goal": [0, 0]},
    ],
}


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        Handler.service = SwarmGrid(str(Path(self.directory.name) / "http.db"))
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.directory.cleanup()

    def call(self, method, path, payload=None, key=None):
        body = json.dumps(payload).encode() if payload is not None else None
        request = Request(f"{self.base}{path}", data=body, method=method)
        request.add_header("Content-Type", "application/json")
        if key:
            request.add_header("Idempotency-Key", key)
        try:
            with urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read())
        except HTTPError as error:
            try:
                return error.code, json.loads(error.read())
            finally:
                error.close()

    def raw(self, method, path, payload=None, key=None):
        body = json.dumps(payload).encode() if payload is not None else None
        request = Request(f"{self.base}{path}", data=body, method=method)
        request.add_header("Content-Type", "application/json")
        if key:
            request.add_header("Idempotency-Key", key)
        with urlopen(request, timeout=10) as response:
            return response.status, response.read()

    def test_health(self):
        self.assertEqual((200, {"status": "ok"}), self.call("GET", "/health"))

    def test_full_planning_flow_over_http(self):
        status, created = self.call("POST", "/maps", MAP, "map-1")
        self.assertEqual(201, status)
        self.assertEqual("grid", created["id"])
        status, mission = self.call("POST", "/missions", MISSION, "mission-1")
        self.assertEqual(201, status)
        self.assertEqual("created", mission["status"])
        status, plan = self.call("POST", "/missions/run-1/plan", {}, "plan-1")
        self.assertEqual(200, status)
        self.assertEqual("solved", plan["status"])
        self.assertEqual(8, plan["makespan"])
        status, stored = self.call("GET", "/missions/run-1")
        self.assertEqual(plan, stored["plan"])
        status, schedule = self.call("GET", "/missions/run-1/schedule")
        self.assertEqual(200, status)
        self.assertEqual(9, len(schedule["timeline"]))
        self.assertEqual(
            {"a": None, "b": [0, 0]}, schedule["timeline"][-1]["positions"]
        )
        self.assertEqual("priority", plan["method"])

    def test_error_envelope_and_status_codes(self):
        status, error = self.call("POST", "/maps", {**MAP, "extra": 1}, "k1")
        self.assertEqual(400, status)
        self.assertEqual("validation_error", error["error"]["code"])
        status, error = self.call("GET", "/maps/missing")
        self.assertEqual(404, status)
        self.assertEqual("not_found", error["error"]["code"])
        status, error = self.call("POST", "/missions", MISSION, "k2")
        self.assertEqual(404, status)
        status, error = self.call("GET", "/nope")
        self.assertEqual(404, status)
        self.assertEqual("not_found", error["error"]["code"])

    def test_replayed_request_is_byte_identical(self):
        first_status, first = self.raw("POST", "/maps", MAP, "map-1")
        second_status, second = self.raw("POST", "/maps", MAP, "map-1")
        self.assertEqual(201, first_status)
        self.assertEqual(201, second_status)
        self.assertEqual(first, second)

    def test_idempotency_and_content_type(self):
        self.call("POST", "/maps", MAP, "map-1")
        status, first = self.call("POST", "/missions", MISSION, "same-key")
        self.assertEqual(201, status)
        status, repeated = self.call("POST", "/missions", MISSION, "same-key")
        self.assertEqual(201, status)
        self.assertEqual(first, repeated)
        request = Request(f"{self.base}/maps", data=json.dumps(MAP).encode(), method="POST")
        request.add_header("Content-Type", "text/plain")
        request.add_header("Idempotency-Key", "k9")
        try:
            with urlopen(request, timeout=10) as response:
                self.fail("expected the request to be rejected")
        except HTTPError as error:
            try:
                self.assertEqual(400, error.code)
            finally:
                error.close()

    def test_unsolvable_mission_returns_conflicts_over_http(self):
        self.call("POST", "/maps", {"id": "lane", "width": 4, "height": 1, "grid": ["...."]}, "k1")
        self.call(
            "POST",
            "/missions",
            {
                "id": "swap",
                "map_id": "lane",
                "agents": [
                    {"id": "a", "start": [0, 0], "goal": [3, 0]},
                    {"id": "b", "start": [3, 0], "goal": [0, 0]},
                ],
            },
            "k2",
        )
        status, plan = self.call("POST", "/missions/swap/plan", {}, "k3")
        self.assertEqual(200, status)
        self.assertEqual("unsolvable", plan["status"])
        self.assertTrue(plan["conflicts"])
        status, error = self.call("GET", "/missions/swap/schedule")
        self.assertEqual(409, status)
        self.assertEqual("conflict", error["error"]["code"])


if __name__ == "__main__":
    unittest.main()
