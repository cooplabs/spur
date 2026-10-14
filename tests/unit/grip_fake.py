"""A fake Dex3-1 hand for the grip tests: joints track their command, per-group stops block them, pads rise if blocked.

Kinematics come from r1mc3.hands.grip (the model's): a group's stop is a predicate on the hand's joint angles (an object
at an aperture, the hand touching itself, a surface), so the scenarios are geometric, not scripted outcomes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from r1mc3.hands import grip as G

DT = 0.01
TAU_RAW = 1e-6  # N m per raw unit


class FakeHand:
    def __init__(self, side: str = "right", q0: Any = None) -> None:
        self.side = side
        self.q = G.preset("open", side) if q0 is None else np.asarray(q0, dtype=np.float64).copy()
        self.cmd = self.q.copy()
        self.kp = np.full(7, 1.5)
        self.stops: dict[str, Callable[[np.ndarray], bool]] = {}
        self.blocked: dict[str, bool] = {}
        self.pad_gain_kpa = 40.0  # kPa per rad of push into a stop (a firm press reads a few kPa)
        self.tip_only = True  # the real pads feel only firm FINGERTIP contact (HW1 B3.2)
        self.pad_off: set[str] = set()  # groups whose pads read nothing (a contact off the pad / a load shift)
        self.temperature = 35.0
        self.valid = True
        self.commands = 0
        self.d = G.close_dir(side)
        self.base = np.full((9, 12), 30000.0)
        for k in range(9):
            for s in (0, 2, 9, 11) if k not in (1, 3, 5) else (3, 6, 8):
                self.base[k, s] = 101325.0
        self.rng = np.random.default_rng(0)

    def command(self, side: str, q: Any, *, dq: Any = None, kp: Any = None, kd: Any = None, tau: Any = None,
                ttl_s: float = 1.0) -> dict[str, Any]:  # fmt: skip
        assert side == self.side
        self.cmd = np.asarray(q, dtype=np.float64).copy()
        if kp is not None:
            self.kp = np.asarray(kp, dtype=np.float64).copy()
        self.commands += 1
        return {"ok": True}

    def step(self) -> None:
        err = self.cmd - self.q
        dq = np.clip(err * 10.0, -3.0, 3.0)
        q_new = self.q + dq * DT
        for g, idx in G.GROUPS.items():
            stop = self.stops.get(g)
            trial = self.q.copy()
            trial[list(idx)] = q_new[list(idx)]
            closing = float(np.max(self.d[list(idx)] * dq[list(idx)])) > 0
            # a collision stays blocked while the command keeps closing (a point-distance predicate would let the
            # tips slide past each other's closest approach)
            if stop is not None and closing and (stop(trial) or self.blocked.get(g, False)):
                q_new[list(idx)] = self.q[list(idx)]
                self.blocked[g] = True
            elif not closing:
                self.blocked[g] = False
        q_new[0] = self.q[0] + np.clip(err[0] * 10.0, -3.0, 3.0) * DT
        self.q = q_new

    def state(self, side: str) -> dict[str, Any]:
        tau = self.kp * (self.cmd - self.q)
        p = self.base.copy() + self.rng.normal(0.0, 50.0, (9, 12)) * (self.base > 50000)
        for g, (prox, tip) in G.GROUP_PADS.items():
            if self.blocked.get(g) and g not in self.pad_off:
                idx = [i for i in G.GROUPS[g]]
                push = max(0.0, float(np.max(self.d[idx] * (self.cmd[idx] - self.q[idx]))))
                for k in (tip,) if self.tip_only else (prox, tip):
                    for s in (3, 6, 8) if k in (1, 3, 5) else (0, 2, 9, 11):
                        p[k, s] += 1000.0 * self.pad_gain_kpa * push
        return {
            "q": self.q.copy(),
            "dq": np.zeros(7),
            "tau_est": tau / TAU_RAW,
            "pressure": p,
            "temperature": np.full((7, 2), self.temperature),
            "valid": self.valid,
        }


def run(runner: G.GraspRunner, hand: FakeHand, t_s: float, t0: float = 0.0,
        until: tuple[str, ...] = ("TENTATIVE", "HOLDING", "NOT_HOLDING")) -> float:  # fmt: skip
    """Tick runner + hand at 100 Hz from t0 for at most t_s or until a state in `until`; returns the time reached."""
    t = t0
    while t < t0 + t_s:
        hand.step()
        t += DT
        st = runner.tick(round(t * 1e9))
        if st in until:
            break
    return t


def aperture_stop(side: str, finger: str, ap_m: float) -> Callable[[np.ndarray], bool]:
    """Block when the thumb tip comes within the tips-met distance + ap_m of the finger tip (an object / a layer)."""
    z = G.empty_aperture(side, finger)

    def f(q: np.ndarray) -> bool:
        t = G.tips(side, q)
        return bool(np.linalg.norm(t["thumb"] - t[finger]) - z <= ap_m)

    return f


def closure_stop(side: str, joints: tuple[int, ...], closure: float) -> Callable[[np.ndarray], bool]:
    """Block when the given joints pass a fraction of the way from open to the fist (a surface under the tips)."""
    qo, qf = G.preset("open", side), G.preset("fist", side)
    d = G.close_dir(side)

    def f(q: np.ndarray) -> bool:
        j = list(joints)
        frac = (d[j] * (q[j] - qo[j])) / np.maximum(d[j] * (qf[j] - qo[j]), 1e-6)
        return bool(np.max(frac) >= closure)

    return f


def layer_stop(side: str, finger: str, layer_m: float) -> Callable[[np.ndarray], bool]:
    """Block when the thumb's and the finger's tip shells come within layer_m (a thin layer between them)."""

    def f(q: np.ndarray) -> bool:
        return bool(G.tip_clearance(side, q, finger) <= layer_m)

    return f
