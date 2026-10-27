"""Claude drives OceanKing on the seabed: find the trash, pick it up, bring it to the zone.

A manual tool-use loop over the Claude API (the simulation steps inside the tools,
so the loop is ours). Claude sees the robot only as the robot sees itself: the
onboard camera's image, what the colour detector makes of it (`perception.py`),
its odometry and its claw. The tools are `skills.Skills`:

    look, turn, walk, approach, grasp, release, go_to, status, finish

    .venv/bin/python -m oceanking.demo.agent --video run.mp4 [--max-turns 60]

Needs credentials for the Claude API (`ANTHROPIC_API_KEY`, or an `ant auth login`
profile); `--dry-run` drives the same loop with a scripted stand-in for Claude
(`ScriptedClient`), which is what the tests use.

Model: Claude Opus 5.5 with adaptive thinking, and the server-side refusal
fallback (`fallbacks: "default"`), which re-runs a declined request on a fallback
model inside the same call.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
#: Claude Opus 5.5 prices, USD per million tokens: input, cache write (5 min),
#: cache read, output. A fallback turn is billed at its own model's rates, which
#: this does not see; it is an estimate for the budget guard, not the bill.
PRICE = {"input": 4.00, "cache_write": 5.00, "cache_read": 0.20, "output": 20.00}
#: Where every run's estimated spend is appended (git-ignored).
LEDGER = Path(__file__).resolve().parents[2] / "scratch" / "api_ledger.jsonl"


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Spend:
    """Accumulates `usage` across requests and stops a run past `limit_usd`."""

    limit_usd: float = 5.0
    input: int = 0
    cache_write: int = 0
    cache_read: int = 0
    output: int = 0

    def add(self, usage) -> None:
        g = lambda k: int(getattr(usage, k, 0) or 0)  # noqa: E731
        self.input += g("input_tokens")
        self.cache_write += g("cache_creation_input_tokens")
        self.cache_read += g("cache_read_input_tokens")
        self.output += g("output_tokens")
        if self.usd > self.limit_usd:
            raise BudgetExceeded(f"estimated spend {self.usd:.2f} USD passed the "
                                 f"{self.limit_usd:.2f} USD limit")

    @property
    def usd(self) -> float:
        return (self.input * PRICE["input"] + self.cache_write * PRICE["cache_write"]
                + self.cache_read * PRICE["cache_read"] + self.output * PRICE["output"]) / 1e6

    def as_dict(self) -> dict:
        return {"input": self.input, "cache_write": self.cache_write,
                "cache_read": self.cache_read, "output": self.output,
                "usd_estimate": round(self.usd, 4)}


def load_key(env_file: Path | None = None) -> None:
    """Put the key from the repository's `.env` (`CLAUDE_API_KEY` or
    `ANTHROPIC_API_KEY`) into `ANTHROPIC_API_KEY`, unless one is set already. The
    file is git-ignored; the key is never printed."""
    import os

    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env_file = env_file or Path(__file__).resolve().parents[2] / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        key, _, value = line.strip().partition("=")
        if key in ("CLAUDE_API_KEY", "ANTHROPIC_API_KEY") and value.strip():
            os.environ["ANTHROPIC_API_KEY"] = value.strip().strip('"').strip("'")
            return

SYSTEM = """You are the onboard controller of OceanKing, a six-legged crab robot walking on \
the seabed. Its left front leg is carried as a claw. Your mission: collect the sunken \
trash (a red can, a green bottle, a yellow carton) one piece at a time and bring each to \
the drop zone, a magenta disk with a magenta pole and flag, then call finish.

What you can sense: `look` returns the onboard camera image (a fisheye-like 96 degree view \
from the robot's front; the robot's own red body and claw appear at the bottom) and a \
detector's list of what it recognised, with ground distance and bearing. Your odometry and \
claw state come from `status`. You do not know where anything is until you have seen it.

How to work:
- Look first. If a target is not in view, turn (e.g. 60 degrees) and look again.
- `approach` a detected piece of trash to put the open claw on it, then `grasp`. If grasp \
reports nothing in the claw, `release`, step back with `walk` (0.15 m back), look and try \
again.
- With trash in the claw, find the drop zone (look/turn), `go_to` it, then `release`.
- The robot walks slowly (about 0.12 m/s). Prefer few, purposeful actions; check with \
`look` after moves that may have changed the view.
- Call `finish` when all trash you can find is in the zone, or if you are stuck after \
several attempts, with a one-line report."""

TOOLS: list[dict[str, Any]] = [
    {"name": "look", "description": "Capture the onboard camera image and run the trash/"
     "drop-zone detector on it. Returns the image and the detections (label, ground distance, "
     "bearing; positive bearing is to the left).",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "turn", "description": "Turn in place by the given angle (positive is left/"
     "counter-clockwise).",
     "input_schema": {"type": "object", "properties": {
         "degrees": {"type": "number", "description": "Angle to turn, -180 to 180."}},
         "required": ["degrees"], "additionalProperties": False}},
    {"name": "walk", "description": "Walk a distance in a body-relative direction, by odometry.",
     "input_schema": {"type": "object", "properties": {
         "metres": {"type": "number", "description": "Distance, 0.05 to 2.0 m."},
         "direction": {"type": "string", "enum": ["forward", "back", "left", "right"]}},
         "required": ["metres", "direction"], "additionalProperties": False}},
    {"name": "approach", "description": "Walk to a piece of trash the detector has seen and "
     "put the open claw on it. Uses the last detection, re-detecting on the way.",
     "input_schema": {"type": "object", "properties": {
         "label": {"type": "string", "enum": ["can", "bottle", "carton"]}},
         "required": ["label"], "additionalProperties": False}},
    {"name": "grasp", "description": "Close the claw and lift. Reports whether something is "
     "held (from how far the claw closed).",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "release", "description": "Open the claw and put the arm back.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "go_to", "description": "Walk until the claw is over the drop zone (it must have "
     "been seen by look).",
     "input_schema": {"type": "object", "properties": {
         "label": {"type": "string", "enum": ["drop_zone"]}},
         "required": ["label"], "additionalProperties": False}},
    {"name": "status", "description": "Odometry, heading, claw state, and the last ground "
     "fixes of everything seen so far.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "finish", "description": "End the mission with a one-line report.",
     "input_schema": {"type": "object", "properties": {
         "report": {"type": "string"}}, "required": ["report"], "additionalProperties": False}},
]


def _png_b64(img) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


@dataclass
class Transcript:
    events: list[dict] = field(default_factory=list)

    def add(self, **kw) -> None:
        self.events.append(kw)


def run_tool(skills, name: str, args: dict) -> list[dict] | str:
    """Execute one tool; returns tool_result content (a string, or blocks with an image)."""
    if name == "look":
        img, dets = skills.look()
        text = ("Detections: " + "; ".join(d.describe() for d in dets)) if dets else \
            "Detections: none of the targets is in view."
        return [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": _png_b64(img)}},
                {"type": "text", "text": text}]
    if name == "turn":
        return skills.turn(float(args["degrees"]))
    if name == "walk":
        metres = min(max(float(args["metres"]), 0.05), 2.0)
        return skills.walk(metres, args["direction"])
    if name == "approach":
        return skills.approach(args["label"])
    if name == "grasp":
        return skills.grasp()
    if name == "release":
        return skills.release()
    if name == "go_to":
        return skills.go_to(args["label"])
    if name == "status":
        return skills.status()
    raise ValueError(f"unknown tool {name}")


def run_mission(client, skills, *, max_turns: int = 60, effort: str = "medium",
                transcript: Transcript | None = None, caption=None,
                spend: Spend | None = None) -> str:
    """The agent loop. `client` is an `anthropic.Anthropic` (or a stand-in with the
    same `beta.messages.create`). Returns the finish report."""
    transcript = transcript or Transcript()
    spend = spend or Spend(limit_usd=float("inf"))
    messages: list[dict] = [{"role": "user", "content":
                             "Start the mission. The robot is standing still, claw empty."}]
    for turn in range(max_turns):
        response = client.beta.messages.create(
            model=MODEL, max_tokens=16000, system=SYSTEM, tools=TOOLS, messages=messages,
            thinking={"type": "adaptive"}, output_config={"effort": effort},
            betas=[FALLBACK_BETA], fallbacks="default",
            # Every turn resends the history: cache the prefix up to its last block.
            cache_control={"type": "ephemeral"},
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            try:
                spend.add(usage)
            except BudgetExceeded as error:
                transcript.add(turn=turn, budget=str(error))
                return f"stopped: {error}"
        if response.stop_reason == "refusal":
            transcript.add(turn=turn, refusal=True)
            return "the request was declined"
        messages.append({"role": "assistant", "content": response.content})
        for block in response.content:
            if getattr(block, "type", None) == "text" and block.text.strip():
                transcript.add(turn=turn, say=block.text)
        calls = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
        if not calls:
            if response.stop_reason == "end_turn":
                messages.append({"role": "user", "content": "Continue the mission, or call "
                                 "finish if it is complete."})
                continue
            continue
        results = []
        for call in calls:
            if call.name == "finish":
                transcript.add(turn=turn, finish=call.input.get("report", ""))
                return call.input.get("report", "")
            if caption is not None:
                caption(f"{call.name} {json.dumps(call.input)}")
            t0 = time.time()
            try:
                content = run_tool(skills, call.name, call.input)
                is_error = False
            except Exception as error:  # noqa: BLE001 - reported back to Claude
                content, is_error = f"error: {error}", True
            summary = content if isinstance(content, str) else content[-1]["text"]
            transcript.add(turn=turn, tool=call.name, input=call.input, result=summary,
                           wall_s=round(time.time() - t0, 1), sim_s=round(skills.sim.time, 1))
            results.append({"type": "tool_result", "tool_use_id": call.id, "content": content,
                            "is_error": is_error})
        messages.append({"role": "user", "content": results})
    return "stopped: turn limit reached"


# ── a scripted stand-in for Claude (tests, --dry-run) ───────────────────────


@dataclass
class _Block:
    type: str
    text: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)
    id: str = ""


@dataclass
class _Response:
    content: list
    stop_reason: str


class ScriptedClient:
    """Answers each request with the next tool call of `script`, a list of
    (tool name, input). Records every request it was sent."""

    def __init__(self, script: list[tuple[str, dict]]):
        self.script = list(script)
        self.requests: list[dict] = []
        outer = self

        class _Messages:
            def create(self, **kw):
                outer.requests.append({**kw, "messages": list(kw["messages"])})
                if not outer.script:
                    return _Response([_Block("tool_use", name="finish", id="f",
                                             input={"report": "script ended"})], "tool_use")
                name, args = outer.script.pop(0)
                return _Response([_Block("text", text=f"next: {name}"),
                                  _Block("tool_use", name=name, input=args,
                                         id=f"t{len(outer.requests)}")], "tool_use")

        class _Beta:
            messages = _Messages()

        self.beta = _Beta()


def score(sim) -> dict:
    """Ground truth after a run: where each piece of trash is relative to the zone."""
    from oceanking.demo.scene import DROP_RADIUS, DROP_ZONE

    out = {}
    for name, p in sim.positions().items():
        d = math.hypot(p[0] - DROP_ZONE[0], p[1] - DROP_ZONE[1])
        out[name] = {"distance_to_zone_m": round(d, 3), "in_zone": d <= DROP_RADIUS + 0.03}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", default=None, help="record an MP4 of the run here")
    ap.add_argument("--transcript", default=None, help="write the transcript (JSON) here")
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--budget-usd", type=float, default=5.0,
                    help="stop the run when its estimated spend passes this (default 5)")
    ap.add_argument("--effort", default="medium", choices=("low", "medium", "high", "xhigh"))
    ap.add_argument("--checkpoint", default=None,
                    help="five-foot policy (default: the arm-out one if trained, else official)")
    ap.add_argument("--arm", default=None, choices=("stow", "reach"),
                    help="the arm the policy walks with (default: from the checkpoint)")
    ap.add_argument("--dry-run", action="store_true",
                    help="no API: a scripted stand-in fetches one piece (--fetch)")
    ap.add_argument("--fetch", default="can", choices=("can", "bottle", "carton"),
                    help="what --dry-run's stand-in fetches")
    args = ap.parse_args()

    from oceanking.demo.sim import CHECKPOINT, CHECKPOINT_REACH, OceanSim
    from oceanking.demo.skills import Skills

    checkpoint = args.checkpoint or (CHECKPOINT_REACH if CHECKPOINT_REACH.exists() else CHECKPOINT)
    arm = args.arm or ("reach" if "reach" in str(checkpoint) else "stow")
    sim = OceanSim(checkpoint=checkpoint, arm=arm)
    skills = Skills(sim, verbose=True)
    if args.video:
        sim.record(args.video)
    if args.dry_run:
        client = ScriptedClient(demo_script(args.fetch))
    else:
        import anthropic

        load_key()
        client = anthropic.Anthropic()
    transcript = Transcript()
    spend = Spend(limit_usd=args.budget_usd)

    def caption(text):
        sim.caption = text

    try:
        report = run_mission(client, skills, max_turns=args.max_turns, effort=args.effort,
                             transcript=transcript, caption=caption, spend=spend)
    finally:
        if not args.dry_run:
            LEDGER.parent.mkdir(parents=True, exist_ok=True)
            with open(LEDGER, "a") as f:
                f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "video": args.video, **spend.as_dict()}) + "\n")
            print(f"[agent] estimated spend this run: {spend.usd:.3f} USD "
                  f"(ledger {LEDGER})")
    result = {"report": report, "score": score(sim), "sim_s": round(sim.time, 1),
              "spend": spend.as_dict(), "events": transcript.events}
    sim.close()
    print(json.dumps({k: result[k] for k in ("report", "score", "sim_s", "spend")}, indent=2))
    if args.transcript:
        with open(args.transcript, "w") as f:
            json.dump(result, f, indent=2, default=str)


def demo_script(label: str = "can") -> list[tuple[str, dict]]:
    """What a scripted stand-in does with `--dry-run`: fetch `label` to the zone (the
    zone lies to the robot's left of every piece's position)."""
    return [("look", {}), ("approach", {"label": label}), ("grasp", {}),
            ("turn", {"degrees": 100}), ("look", {}), ("go_to", {"label": "drop_zone"}),
            ("release", {}), ("status", {})]


if __name__ == "__main__":
    main()
