from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import ValidationError

MAX_IDENTIFIER = 100
MAX_DIMENSION = 64
MAX_CELLS = 4096
MAX_AGENTS = 32

Point = tuple[int, int]


def identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_IDENTIFIER:
        raise ValidationError(f"{field} must be a non-empty string of at most {MAX_IDENTIFIER} characters")
    if value != value.strip():
        raise ValidationError(f"{field} must not have leading or trailing whitespace")
    return value


def integer(value: Any, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field} must be an integer")
    if value < minimum or value > maximum:
        raise ValidationError(f"{field} must be between {minimum} and {maximum}")
    return value


def point(value: Any, field: str, width: int, height: int) -> Point:
    if not isinstance(value, list) or len(value) != 2:
        raise ValidationError(f"{field} must be a [x, y] pair")
    x = integer(value[0], f"{field}[0]", 0, width - 1)
    y = integer(value[1], f"{field}[1]", 0, height - 1)
    return (x, y)


def decode_point(value: Any) -> Point:
    return (int(value[0]), int(value[1]))


@dataclass(frozen=True)
class Map:
    """A rectangular 4-connected grid; `obstacle` marks cells agents may never enter."""

    id: str
    width: int
    height: int
    rows: tuple[str, ...]
    obstacle: str

    @classmethod
    def parse(cls, raw: Any) -> "Map":
        if not isinstance(raw, dict):
            raise ValidationError("map must be an object")
        allowed = {"id", "width", "height", "grid", "obstacle"}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValidationError(f"map contains unknown fields: {', '.join(unknown)}")
        required = {"id", "width", "height", "grid"}
        missing = sorted(required - set(raw))
        if missing:
            raise ValidationError(f"map is missing required fields: {', '.join(missing)}")
        map_id = identifier(raw["id"], "map id")
        width = integer(raw["width"], "width", 1, MAX_DIMENSION)
        height = integer(raw["height"], "height", 1, MAX_DIMENSION)
        if width * height > MAX_CELLS:
            raise ValidationError(f"map may not exceed {MAX_CELLS} cells")
        if "grid" in raw and isinstance(raw["grid"], list) and len(raw["grid"]) != height:
            raise ValidationError(f"grid must be an array of exactly {height} row strings")
        obstacle = raw.get("obstacle", "#")
        if not isinstance(obstacle, str) or len(obstacle) != 1 or obstacle.isspace():
            raise ValidationError("obstacle must be a single non-whitespace character")
        grid = raw["grid"]
        if not isinstance(grid, list) or len(grid) != height:
            raise ValidationError(f"grid must be an array of exactly {height} row strings")
        rows: list[str] = []
        for index, row in enumerate(grid):
            if not isinstance(row, str):
                raise ValidationError(f"grid row {index} must be a string")
            if len(row) != width:
                raise ValidationError(f"grid row {index} must be exactly {width} characters long")
            for column, char in enumerate(row):
                if char not in (".", obstacle):
                    raise ValidationError(
                        f"grid row {index} column {column} must be '.' or the obstacle character {obstacle!r}"
                    )
            rows.append(row)
        return cls(map_id, width, height, tuple(rows), obstacle)

    def walkable(self, cell: Point) -> bool:
        x, y = cell
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return False
        return self.rows[y][x] != self.obstacle

    def index(self, cell: Point) -> int:
        return cell[1] * self.width + cell[0]

    def neighbors(self, cell: Point) -> tuple[Point, ...]:
        x, y = cell
        candidates = ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1))
        return tuple(candidate for candidate in candidates if self.walkable(candidate))

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "width": self.width,
            "height": self.height,
            "grid": list(self.rows),
            "obstacle": self.obstacle,
        }


@dataclass(frozen=True)
class Agent:
    id: str
    start: Point
    goal: Point

    @classmethod
    def parse(cls, raw: Any, map_record: Map, field: str) -> "Agent":
        if not isinstance(raw, dict):
            raise ValidationError(f"{field} must be an object")
        unknown = sorted(set(raw) - {"id", "start", "goal"})
        if unknown:
            raise ValidationError(f"{field} contains unknown fields: {', '.join(unknown)}")
        missing = sorted({"id", "start", "goal"} - set(raw))
        if missing:
            raise ValidationError(f"{field} is missing required fields: {', '.join(missing)}")
        agent_id = identifier(raw["id"], f"{field}.id")
        start = point(raw["start"], f"{field}.start", map_record.width, map_record.height)
        goal = point(raw["goal"], f"{field}.goal", map_record.width, map_record.height)
        if not map_record.walkable(start):
            raise ValidationError(f"{field}.start {list(start)} is an obstacle cell")
        if not map_record.walkable(goal):
            raise ValidationError(f"{field}.goal {list(goal)} is an obstacle cell")
        return cls(agent_id, start, goal)

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "start": list(self.start), "goal": list(self.goal)}


@dataclass(frozen=True)
class Mission:
    id: str
    map_id: str
    agents: tuple[Agent, ...]

    @classmethod
    def parse(cls, raw: Any, map_record: Map) -> "Mission":
        if not isinstance(raw, dict):
            raise ValidationError("mission must be an object")
        unknown = sorted(set(raw) - {"id", "map_id", "agents"})
        if unknown:
            raise ValidationError(f"mission contains unknown fields: {', '.join(unknown)}")
        missing = sorted({"id", "map_id", "agents"} - set(raw))
        if missing:
            raise ValidationError(f"mission is missing required fields: {', '.join(missing)}")
        mission_id = identifier(raw["id"], "mission id")
        map_id = identifier(raw["map_id"], "map_id")
        raw_agents = raw["agents"]
        if not isinstance(raw_agents, list) or not raw_agents:
            raise ValidationError("agents must be a non-empty array")
        if len(raw_agents) > MAX_AGENTS:
            raise ValidationError(f"a mission may not exceed {MAX_AGENTS} agents")
        agents = tuple(
            Agent.parse(entry, map_record, f"agents[{index}]") for index, entry in enumerate(raw_agents)
        )
        ids = [agent.id for agent in agents]
        if len(set(ids)) != len(ids):
            raise ValidationError("agent ids must be unique within a mission")
        starts = [agent.start for agent in agents]
        if len(set(starts)) != len(starts):
            raise ValidationError("agent start cells must be unique: two agents may not start on one cell")
        for agent in agents:
            if agent.start == agent.goal:
                raise ValidationError(f"agent {agent.id} already starts on its goal cell")
        return cls(mission_id, map_id, agents)

    def order(self, priority: list[str]) -> tuple[Agent, ...]:
        by_id = {agent.id: agent for agent in self.agents}
        return tuple(by_id[agent_id] for agent_id in priority)

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "map_id": self.map_id, "agents": [agent.as_dict() for agent in self.agents]}
