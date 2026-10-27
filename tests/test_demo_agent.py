"""The demo agent's loop, tools and perception, without the Claude API.

Silent failures pinned here: a request that is not what the API expects (model,
tools, the refusal fallback), a `look` whose image never reaches Claude, a tool
error that ends the loop instead of being reported back, a detector that finds
the robot's own red body or misses the trash, and a turn that does not turn.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

tasks = pytest.importorskip("tasks")
pytest.importorskip("mjlab")


@pytest.fixture(scope="module")
def skills():
    from oceanking.demo.sim import OceanSim
    from oceanking.demo.skills import Skills

    sim = OceanSim()
    yield Skills(sim)
    sim.close()


def test_detector_finds_the_trash_where_it_lies_and_not_the_robot(skills):
    """From the spawn the three pieces are in view (control: the robot's own red
    body fills the bottom of the image and must not be reported as the can)."""
    from oceanking.demo.perception import detect

    sim = skills.sim
    _, dets = skills.look()
    truth = sim.positions()
    found = {d.label: d for d in dets if d.ground is not None and d.whole}
    # The carton sits at the image's edge from the spawn: cut there, its fix is not
    # trusted (`whole` False), which is the point of the flag.
    assert not any(d.label == "carton" and d.whole for d in dets)
    # The bottle is in plain view: its fix is within 10 cm. The can stands behind
    # the stowed claw from here, which hides its foot: it is reported, flagged
    # approximate, not trusted.
    assert "bottle" in found, f"bottle not detected: {[d.describe() for d in dets]}"
    assert np.hypot(*(np.array(found["bottle"].ground) - truth["bottle"][:2])) < 0.10
    assert any(d.label == "can" for d in dets)
    img = sim.camera()
    pos, mat = sim.camera_pose()
    unmasked = detect(img, sim.camera_fov()[1], pos, mat, sim.pose())
    assert max(d.pixels for d in unmasked if d.label == "can") > 5 * max(
        (d.pixels for d in dets if d.label == "can"), default=1), "the self-mask did nothing"


def test_turn_turns(skills):
    before = skills.sim.pose()[2]
    skills.turn(45)
    turned = math.degrees(math.atan2(math.sin(skills.sim.pose()[2] - before),
                                     math.cos(skills.sim.pose()[2] - before)))
    assert 30 < turned < 60
    skills.turn(-45)


def test_the_loop_sends_valid_requests_images_and_errors(skills):
    from oceanking.demo import agent

    client = agent.ScriptedClient([("look", {}), ("approach", {"label": "nothing"}),
                                   ("status", {}), ("finish", {"report": "done"})])
    transcript = agent.Transcript()
    report = agent.run_mission(client, skills, max_turns=10, transcript=transcript)
    assert report == "done"
    first = client.requests[0]
    assert first["model"] == agent.MODEL
    assert first["fallbacks"] == "default" and agent.FALLBACK_BETA in first["betas"]
    assert first["thinking"] == {"type": "adaptive"}
    assert {t["name"] for t in first["tools"]} >= {"look", "approach", "grasp", "finish"}
    # The look's result carries the image, as a base64 PNG block.
    second = client.requests[1]["messages"][-1]["content"][0]
    assert second["type"] == "tool_result"
    image = second["content"][0]
    assert image["type"] == "image" and image["source"]["media_type"] == "image/png"
    assert len(image["source"]["data"]) > 1000
    # A failing tool is reported back, not raised: the loop went on to `status`.
    third = client.requests[2]["messages"][-1]["content"][0]
    assert "cannot approach nothing" in str(third["content"]) or third["is_error"]
    assert [e.get("tool") for e in transcript.events if "tool" in e] == ["look", "approach",
                                                                         "status"]
