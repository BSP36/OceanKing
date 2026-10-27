"""The crown's ballast tank, driven while the simulation runs (docs/PHYSICS.md, crown).

The crown displaces its outer volume whatever is in it; what the tank changes is
its **mass**: `crown_tank_volume` of water when flooded, none when blown. A pump
moves the water at a finite rate, so the air fraction follows its target at
`rate` per second. Each change rewrites the crown body's mass, centre of mass
(the water settles at the tank floor) and inertia, and the per-environment water
state the observations read (`OceanState.beta`, net weight, load scale), so a
policy observing beta sees the tank's effect.

    cfg.events["crown_tank"] = EventTermCfg(func=CrownTank, mode="step",
        params={"hydro": base.as_dict(), "rate": 0.5, "script": "tank.py"})

`script` is a Python file defining `air(t) -> float in [0, 1]` (seconds since the
episode began), or `air(t, state)` with `state` the robot's base height `z` (m,
above the env origin) and vertical speed `vz` (m/s), for feedback such as an
altitude hold; without one the target stays at `start_air`. `set_target` sets it
from code (the demo agent).
"""

from __future__ import annotations

import importlib.util
from typing import Any

import torch
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

from oceanking import crown
from oceanking.buoyancy import crown_tank_volume
from oceanking.hydro import HydroCfg
from oceanking.water import G, KEEL, ocean_state

CROWN = "robot/oceanking_crown"


def load_script(path: str):
    spec = importlib.util.spec_from_file_location("oceanking_tank_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "air", None)):
        raise ValueError(f"{path} defines no air(t)")
    import inspect

    fn = mod.air
    if len(inspect.signature(fn).parameters) >= 2:
        return fn
    return lambda t, state: fn(t)


@requires_model_fields("body_mass", "body_inertia", "body_ipos",
                       recompute=RecomputeLevel.set_const)
class CrownTank:
    """Pump the crown's tank towards its target air fraction (step event)."""

    def __init__(self, cfg: Any, env: Any) -> None:
        p = cfg.params
        base = HydroCfg(**p["hydro"])
        ballast = base.ballast()
        self._rho = base.water_density
        self._volume = ballast.total_volume
        self._tank = crown_tank_volume()
        self._shell = ballast.crown_shell_mass
        # Everything but the keel and the tank's water: the robot and the dry crown.
        self._dry = ballast.robot_mass + ballast.crown_shell_mass
        self._land_weight = ballast.robot_mass * G
        m = env.sim.mj_model
        self._crown = m.body(CROWN).id
        self._keel = m.body(KEEL).id
        n = env.num_envs
        self.air = torch.full((n,), float(p.get("start_air", base.crown_air)), device=env.device)
        self.target = self.air.clone()
        self._script = load_script(p["script"]) if p.get("script") else None
        self._step_dt = float(env.step_dt)
        self._dirty = True
        env.crown_tank = self

    def set_target(self, air: float | torch.Tensor) -> None:
        self.target[:] = torch.as_tensor(air, device=self.target.device).clamp(0.0, 1.0)

    def __call__(self, env: Any, env_ids: Any, hydro: dict, rate: float = 0.5,
                 start_air: float | None = None, script: str | None = None) -> None:
        if self._script is not None:
            t = env.episode_length_buf.float() * self._step_dt
            data = env.scene["robot"].data
            z = (data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]).tolist()
            vz = data.root_link_lin_vel_w[:, 2].tolist()
            self.target[:] = torch.tensor(
                [float(self._script(float(x), {"z": z[i], "vz": vz[i]})) for i, x in enumerate(t)],
                device=self.target.device).clamp(0.0, 1.0)
        # A new episode starts from `start_air` (the reset put the robot back too).
        fresh = env.episode_length_buf <= 1
        if fresh.any() and start_air is not None:
            self.air[fresh] = float(start_air)
            self._dirty = True
        step = rate * self._step_dt
        new = self.air + (self.target - self.air).clamp(-step, step)
        if not self._dirty and torch.equal(new, self.air):
            return
        self.air = new
        self._dirty = False
        self.write(env)

    def write(self, env: Any) -> None:
        """The crown's inertial and the observed water state, from `self.air`."""
        model = env.sim.model
        water = self._rho * self._tank * (1.0 - self.air)
        mass = self._shell + water
        model.body_mass[:, self._crown] = mass.to(model.body_mass.dtype)
        heights = torch.tensor([crown.centre_height(float(w), self._rho) for w in water],
                               device=water.device, dtype=model.body_ipos.dtype)
        model.body_ipos[:, self._crown, 2] = heights
        inertia = torch.tensor([crown.inertia(float(mm)) for mm in mass],
                               device=water.device, dtype=model.body_inertia.dtype)
        model.body_inertia[:, self._crown] = inertia
        keel = model.body_mass[:, self._keel].to(water.dtype)
        total = self._dry + water + keel
        state = ocean_state(env)
        state.beta[:] = self._rho * self._volume / total
        state.net_weight[:] = (total - self._rho * self._volume) * G
        state.load_scale[:] = state.net_weight / self._land_weight


def beta_at(hydro: dict, air: float) -> float:
    """Overall beta of the kit `hydro` with the tank at `air`."""
    return HydroCfg(**hydro).ballast().beta_at(air)


__all__ = ["CrownTank", "beta_at", "load_script"]
