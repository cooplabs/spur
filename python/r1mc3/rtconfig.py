"""The r1rt / r1guard processes' configuration file (session 5; rt/include/r1rt/proc_config.hpp is the reader).

    python -m r1mc3.rtconfig --out FILE [--writer-hz 500] [--no-guard-gate] [--weights PATH] [--observe-only]
    python -m r1mc3.rtconfig --out FILE --profile robot [--profile-path YAML] [--weights PATH] [--observe-only]

One source of numbers: the file is the SAME dict `r1sim.v3_core.core_config(...)` builds for the lockstep simulator
(the r1rt_py.Core constructor's keys: limits, gains, the policy interface from the ONNX metadata, the CoM-centred
stand pose of the real-profile model, the takeover gates), flattened to `key = value` lines (`policy.` for the policy
interface, `guard.` for r1guard's parameters), so the processes and the lockstep tests run identical values; float
values are written with Python's repr (round-trips exactly through C's strtod).

Differences from the lockstep default, on purpose (the processes are the robot's arrangement, DESIGN §2.1 / §4.4):
`takeover_require_guard = 1` (a takeover is refused without a fresh r1guard heartbeat; `--no-guard-gate` only for a
single-process test), and the guard section (1 kHz, stale after 5 writer periods, 0.3 s blend from a takeover mode).

The C++ reader refuses any missing / unknown key, so the two sides cannot drift silently (tests run
`r1rt --check-config` on files written here).

**The robot's file (session 6): `--profile robot`** (`robot_config()`): the values come from the robot profile
(`config/robot_profile.r1_dex3.yaml`, version >= 2: rates, limits, gains, safety, executor, processes - each with its
source and date), `takeover_require_rt = 1` and the guard gate on, and every written key's provenance goes into
`<file>.provenance.json` (+ a comment line above each key in the file itself). What stays the repository's / the
model's (`ROBOT_KEPT`): the policy interface and weights (the profile's policy / arm gains are CHECKED against the
ONNX metadata - a mismatch refuses, the policy was trained with them) and the stand pose (the model's CoM-centred
stand until HW1 measures it). `--profile sim` (the default) is unchanged: the simulator's numbers, the tests' file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

FORMAT = "r1rt-config/1"
DEFAULT_PROFILE = Path(__file__).resolve().parents[2] / "config" / "robot_profile.r1_dex3.yaml"

GUARD_DEFAULTS: dict[str, Any] = {
    "cycle_rate_hz": 1000,  # DESIGN §2.1: the heartbeat check at 1 kHz
    "stale_periods": 5,  # 10 ms at 500 Hz (shm_layout.h kHeartbeatStalePeriods)
    "takeover_blend_s": 0.3,  # GuardParams default (guard.hpp)
    "takeover_enabled": 1,
}


def core_dict(
    *,
    writer_hz: int = 500,
    policy_hz: int = 50,
    require_guard: bool = True,
    require_rt: bool = False,
    weights: str | Path | None = None,
    stand_q: Sequence[float] | None = None,
) -> dict[str, Any]:
    """The lockstep simulator's `core_config` dict for the real-profile model, with the process overrides."""
    import numpy as np
    from r1sim import world
    from r1sim.cpp_policy import weights_for
    from r1sim.model import centred_stand_pose
    from r1sim.v3_core import core_config

    from r1mc3.policy import POLICY_DIR, ReferencePolicy

    pol = ReferencePolicy.load(POLICY_DIR)
    if stand_q is None:
        w = world.build_world("real", with_scene=False, visual=False)
        sq = centred_stand_pose(w.model)
    else:
        sq = np.asarray(stand_q, dtype=np.float64)
    wpath = Path(weights) if weights is not None else weights_for(POLICY_DIR / "policy.onnx")
    cfg = core_config(pol, weights=wpath, stand_q=sq, writer_hz=writer_hz, policy_hz=policy_hz)
    cfg["takeover_require_guard"] = 1 if require_guard else 0
    # ABI 5: 1 = TAKEOVER / ON_STAND refused while the process lacks its real-time grants (the robot: --require-rt);
    # 0 here by default because this file's numbers are the simulator's and macOS never grants real time.
    cfg["takeover_require_rt"] = 1 if require_rt else 0
    # ABI 5: ON_STAND's legs + waist hold gains; all 0 = the stand gains (step_api.h) - the lockstep binding's default
    # when core_config leaves them out. The file names every key (proc_config.cpp refuses a missing one).
    cfg.setdefault("on_stand_kp", [0.0] * 26)
    cfg.setdefault("on_stand_kd", [0.0] * 26)
    # ABI 8 (session 9, U19 / D25; docs/sim/STATE_AGE.md): the state-age limit, written explicitly (= the core's
    # defaults) so the robot profile's file and this one agree; the rung OFF (the processes' tests arm it themselves)
    cfg.setdefault("state_age_limit_ns", 10_000_000)
    cfg.setdefault("state_age_sustain_ns", 100_000_000)  # the profile's (STATE_AGE.md); the core's built-in default: 60
    cfg.setdefault("stale_return_live_ns", 500_000_000)
    cfg.setdefault("stale_rung", 0)
    return cfg


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError(f"non-finite value {v!r}")
        return repr(v)
    if isinstance(v, str):
        return v
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, list | tuple):
        return " ".join(_fmt(x) for x in v)
    raise TypeError(f"cannot write {type(v).__name__}")


def lines(cfg: Mapping[str, Any], guard: Mapping[str, Any] | None = None) -> list[str]:
    """The file's lines (format first, then every key; the policy dict and the guard parameters prefixed)."""
    g = dict(GUARD_DEFAULTS)
    if guard:
        unknown = set(guard) - set(GUARD_DEFAULTS)
        if unknown:
            raise KeyError(f"unknown guard parameters: {sorted(unknown)}")
        g.update(guard)
    out = [f"format = {FORMAT}"]
    for k, v in cfg.items():
        if k == "policy":
            for pk, pv in v.items():
                out.append(f"policy.{pk} = {_fmt(pv)}")
        elif k == "weights":
            out.append(f"weights = {'none' if v is None else str(v)}")
        elif k == "writer_hz" or k == "policy_hz":
            out.append(f"{k} = {int(v)}")
        else:
            out.append(f"{k} = {_fmt(v)}")
    for k, v in g.items():
        out.append(f"guard.{k} = {_fmt(v)}")
    return out


def write(path: str | Path, cfg: Mapping[str, Any], guard: Mapping[str, Any] | None = None) -> Path:
    p = Path(path)
    p.write_text(
        "# r1rt / r1guard configuration (python -m r1mc3.rtconfig; read by rt/src/proc_config.cpp)\n"
        + "\n".join(lines(cfg, guard))
        + "\n"
    )
    return p


# ---------------------------------------------------------------------------------------------------------------------
# the robot's file from the robot profile (session 6)

#: Keys of the robot's file that do NOT come from the profile, and why (the profile cannot hold them yet / derived).
ROBOT_KEPT: dict[str, str] = {
    "weights": "the policy weights converted from the repository's policies/*/policy.onnx "
    "(r1sim.cpp_policy.weights_for: the conversion is checked by the ONNX file's sha256)",
    "policy.": "the policy interface from the repository's ONNX metadata (observation terms, joint order, default "
    "pose, action scale, command / lean / waist ranges); its kp / kd for the legs, waist and arms are CHECKED against "
    "the profile's gains.policy_legs_waist / gains.arms_manip (a mismatch refuses: the policy was trained with them)",
    "stand_q": "the CoM-centred stand of the real-profile MODEL (r1sim.model.centred_stand_pose; the 1.6 kg rear mass "
    "of session 5): until HW1 measures the built-in's stand pose and the CoM (HW1_RUNBOOK_ADDITIONS S1, U6)",
    "loop_gap_ns": "derived: 2 writer periods (DESIGN §4.2 loop timing: a gap > 2 periods is a loop_gap event)",
    "takeover_require_rt": "DESIGN §2.1: on the robot a TAKEOVER / ON_STAND is refused without real-time grants "
    "(v2 fell back silently, RES §5) - always 1 in the robot's file",
    "takeover_require_guard": "DESIGN §4.4: no takeover without a live r1guard - always 1 in the robot's file",
    "guard.takeover_enabled": "1 = r1guard takes the token on a stale core (DESIGN §4.3 rung 2); 0 only with "
    "--observe-only (the stand session's board-hold test, explicit OK)",
}


def _prov(section: str, name: str, entry: Any, *, how: str = "") -> dict[str, Any]:
    d: dict[str, Any] = {
        "from": "profile",
        "path": f"{section}.{name}",
        "status": entry.status,
        "source": entry.source,
        "date": str(entry.date),
    }
    if entry.until:
        d["until"] = entry.until
    if how:
        d["how"] = how
    return d


def robot_config(
    profile_path: str | Path = DEFAULT_PROFILE,
    *,
    weights: str | Path | None = None,
    stand_q: Sequence[float] | None = None,
    observe_only: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    """(core dict, guard parameters, provenance per written key) of the robot's processes config.

    Starts from `core_dict` (for the policy interface, the weights and the stand pose - `ROBOT_KEPT`) and replaces
    every value the robot profile holds with the profile's (per joint by name, in the SDK order of `joint_map`).
    Raises ValueError when the profile lacks the `processes` section (version 1), its joint order differs from the
    controller's, or its policy / arm gains differ from the policy's metadata."""
    from r1sim import joints as J

    from r1mc3.config import load_profile

    path = Path(profile_path)
    prof = load_profile(path)
    if prof.processes is None:
        raise ValueError(f"{path}: profile version {prof.version} has no `processes` section (needs version >= 2)")
    names = prof.joint_names
    if tuple(names) != tuple(J.JOINTS):
        raise ValueError(f"{path}: joint_map order differs from the controller's 26 SDK joints")
    r, lim, g, sf, ex, pr = prof.rates, prof.limits, prof.gains, prof.safety, prof.executor, prof.processes
    writer_hz, policy_hz = int(r.writer_hz.value), int(r.policy_hz.value)
    cfg = core_dict(
        writer_hz=writer_hz, policy_hz=policy_hz, require_guard=True, require_rt=True, weights=weights, stand_q=stand_q
    )
    prov: dict[str, dict[str, Any]] = {}

    def vec(m: Any) -> list[float]:
        return [float(m[n]) for n in names]

    def gains_of(*sets: str) -> tuple[list[float], list[float]]:
        """kp, kd per joint: the later sets override the earlier ones where they name the joint."""
        kp = [math.nan] * len(names)
        kd = [math.nan] * len(names)
        for gs in sets:
            for i, n in enumerate(names):
                if n in g[gs].value:
                    kp[i], kd[i] = (float(x) for x in g[gs].value[n])
        if any(math.isnan(x) for x in kp + kd):
            raise ValueError(f"gains {sets}: not every joint has a gain")
        return kp, kd

    # rates
    cfg["writer_hz"], cfg["policy_hz"] = writer_hz, policy_hz
    prov["writer_hz"] = _prov("rates", "writer_hz", r.writer_hz)
    prov["policy_hz"] = _prov("rates", "policy_hz", r.policy_hz)
    cfg["loop_gap_ns"] = round(2e9 / writer_hz)
    # limits
    cfg["q_min"] = [float(lim.position.value[n][0]) for n in names]
    cfg["q_max"] = [float(lim.position.value[n][1]) for n in names]
    prov["q_min"] = prov["q_max"] = _prov("limits", "position", lim.position)
    frac = float(lim.torque_clip_fraction.value)
    cfg["tau_max"] = [frac * x for x in vec(lim.effort.value)]
    prov["tau_max"] = _prov("limits", "effort", lim.effort, how=f"x limits.torque_clip_fraction {frac}")
    prov["tau_max"]["also"] = _prov("limits", "torque_clip_fraction", lim.torque_clip_fraction)
    cfg["dq_max"] = vec(lim.velocity.value)
    prov["dq_max"] = _prov("limits", "velocity", lim.velocity)
    kp_hi = [0.0] * len(names)
    kd_hi = [0.0] * len(names)
    for gs in g.values():
        for i, n in enumerate(names):
            if n in gs.value:
                kp_hi[i] = max(kp_hi[i], float(gs.kp_bounds.max))
                kd_hi[i] = max(kd_hi[i], float(gs.kd_bounds.max))
    cfg["kp_max"], cfg["kd_max"] = kp_hi, kd_hi
    prov["kp_max"] = prov["kd_max"] = {
        "from": "profile",
        "path": "gains.*.kp_bounds / kd_bounds",
        "how": "per joint the largest bound of the gain sets that name it",
        "status": "modelling_choice",
        "source": "the gain sets' bounds (config/robot_profile.r1_dex3.yaml gains)",
        "date": str(g["stiff"].date),
    }
    # gains: the stand / ON_STAND hold (stiff), arms + head HOLD (arms_manip over stiff), damp
    cfg["stand_kp"], cfg["stand_kd"] = gains_of("stiff")
    prov["stand_kp"] = prov["stand_kd"] = _prov("gains", "stiff", g["stiff"])
    cfg["upper_kp"], cfg["upper_kd"] = gains_of("stiff", "arms_manip")
    prov["upper_kp"] = prov["upper_kd"] = _prov(
        "gains", "arms_manip", g["arms_manip"], how="arms from gains.arms_manip, the rest from gains.stiff"
    )
    prov["upper_kp"]["also"] = _prov("gains", "stiff", g["stiff"])
    cfg["damp_kd"] = vec(pr.damp_kd.value)
    prov["damp_kd"] = _prov("processes", "damp_kd", pr.damp_kd)
    if pr.on_stand_hold_gains.value != "stiff":  # the only hold the schema allows today
        raise ValueError(f"processes.on_stand_hold_gains {pr.on_stand_hold_gains.value!r}: only 'stiff' is supported")
    cfg["on_stand_kp"] = [0.0] * len(names)  # all 0 = the stand gains (step_api.h), = gains.stiff above
    cfg["on_stand_kd"] = [0.0] * len(names)
    prov["on_stand_kp"] = prov["on_stand_kd"] = _prov(
        "processes", "on_stand_hold_gains", pr.on_stand_hold_gains, how="all 0 = the stand gains (gains.stiff)"
    )
    # the policy's gains must be the profile's (the policy was trained with them)
    pol = cfg["policy"]
    want_kp, want_kd = gains_of("stiff", "arms_manip", "policy_legs_waist")
    for k, j in enumerate(pol["joint_index"]):
        if abs(float(pol["kp"][k]) - want_kp[j]) > 5e-4 or abs(float(pol["kd"][k]) - want_kd[j]) > 5e-4:
            raise ValueError(
                f"{names[j]}: the policy's kp / kd {pol['kp'][k]} / {pol['kd'][k]} differ from the profile's "
                f"{want_kp[j]} / {want_kd[j]} (the policy was trained with its gains: retrain or fix the profile)"
            )
    # safety (DESIGN §4.2-4.4)
    ms = 1_000_000
    for key, name, conv in (
        ("lowstate_late_ns", "lowstate_age_count_ms", lambda v: round(v * ms)),
        ("lowstate_stale_ns", "lowstate_age_resend_ms", lambda v: round(v * ms)),
        ("lowstate_lost_ns", "lowstate_age_fault_ms", lambda v: round(v * ms)),
        ("policy_max_age_ns", "policy_age_inline_ms", lambda v: round(v * ms)),
        ("fall_tilt_rad", "fall_detect_deg", math.radians),
        ("arm_tilt_stop_rad", "arm_tilt_stop_deg", math.radians),
        ("takeover_tilt_max_rad", "takeover_max_tilt_deg", math.radians),
        ("takeover_dq_max", "takeover_max_dq_rad_s", float),
        ("takeover_temp_max_c", "takeover_max_temp_c", float),
        ("takeover_state_max_age_ns", "takeover_max_state_age_ms", lambda v: round(v * ms)),
    ):
        entry = getattr(sf, name)
        cfg[key] = conv(float(entry.value))
        prov[key] = _prov("safety", name, entry)
    # session 9 (U19 / D25): the state-age limit + the stale rung - written only when the profile has them (absent =
    # the core's defaults, the rung OFF); the rung stays 0 in the shipped profile until the user's OK
    for key, name, conv in (
        ("state_age_limit_ns", "state_age_limit_ms", lambda v: round(v * ms)),
        ("state_age_sustain_ns", "state_age_sustain_ms", lambda v: round(v * ms)),
        ("stale_return_live_ns", "stale_return_live_ms", lambda v: round(v * ms)),
        ("stale_rung", "stale_rung", lambda v: round(v)),
    ):
        entry = getattr(sf, name)
        if entry is not None:
            cfg[key] = conv(float(entry.value))
            prov[key] = _prov("safety", name, entry)
    cfg["stop_time_s"] = float(ex.trajectory_stop_time_s.value)
    prov["stop_time_s"] = _prov("executor", "trajectory_stop_time_s", ex.trajectory_stop_time_s)
    # processes (session 6)
    for key, name, conv in (
        ("stand_ramp_s", "stand_freeze_ramp_s", float),
        ("takeover_blend_s", "takeover_blend_stand_s", float),
        ("action_abs_max", "action_abs_max", float),
        ("builtin_max_age_ns", "builtin_max_age_ms", lambda v: round(v * ms)),
        ("builtin_min_leg_kp", "builtin_min_leg_kp", float),
        ("mirror_max_dev_rad", "mirror_max_dev_rad", float),
        ("takeover_mirror_alarm_ns", "takeover_mirror_alarm_s", lambda v: round(v * 1e9)),
        ("guard_max_age_ns", "guard_max_age_ms", lambda v: round(v * ms)),
    ):
        entry = getattr(pr, name)
        cfg[key] = conv(float(entry.value))
        prov[key] = _prov("processes", name, entry)
    guard = {
        "cycle_rate_hz": int(pr.guard_cycle_hz.value),
        "stale_periods": int(sf.guard_heartbeat_writer_periods.value),
        "takeover_blend_s": float(pr.guard_takeover_blend_s.value),
        "takeover_enabled": 0 if observe_only else 1,
    }
    prov["guard.cycle_rate_hz"] = _prov("processes", "guard_cycle_hz", pr.guard_cycle_hz)
    prov["guard.stale_periods"] = _prov("safety", "guard_heartbeat_writer_periods", sf.guard_heartbeat_writer_periods)
    prov["guard.takeover_blend_s"] = _prov("processes", "guard_takeover_blend_s", pr.guard_takeover_blend_s)
    # never looser than the simulator's tested file without the user's OK (the contract's rule): per joint, a profile
    # limit beyond the simulator's value keeps the simulator's and says so (e.g. the URDF's 33.4 rad/s on the shoulder
    # yaw / elbow / wrist / head vs the core's tested 30 rad/s dq clamp)
    sim_cfg = core_dict()
    for key, tighter in (("q_min", max), ("q_max", min), ("dq_max", min), ("tau_max", min), ("kp_max", min),
                         ("kd_max", min)):  # fmt: skip
        kept = [names[i] for i, (a, b) in enumerate(zip(cfg[key], sim_cfg[key], strict=True)) if tighter(a, b) != a]
        if kept:
            cfg[key] = [tighter(a, b) for a, b in zip(cfg[key], sim_cfg[key], strict=True)]
            prov[key] = dict(prov[key])
            prov[key]["kept_simulator_value_where_looser"] = kept
            prov[key]["how"] = (prov[key].get("how", "") + "; " if prov[key].get("how") else "") + (
                "the simulator's tested value kept where the profile's is looser (loosening needs the user's OK)"
            )
    # what stays the repository's / the model's, or is fixed for the robot
    for key in list(cfg):
        if key == "policy":
            for pk in cfg["policy"]:
                prov[f"policy.{pk}"] = {"from": "repository", "why": ROBOT_KEPT["policy."]}
        elif key not in prov:
            if key not in ROBOT_KEPT:
                raise AssertionError(f"robot config key {key!r} has no provenance (profile or ROBOT_KEPT)")
            prov[key] = {"from": "repository" if key in ("weights", "stand_q") else "fixed", "why": ROBOT_KEPT[key]}
    prov["guard.takeover_enabled"] = {"from": "fixed", "why": ROBOT_KEPT["guard.takeover_enabled"]}
    prov["_profile"] = {
        "path": str(path),
        "name": prof.name,
        "version": prof.version,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return cfg, guard, prov


def differences(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
    """Keys whose written values differ between two config dicts (as the file would write them)."""
    la = dict(x.split(" = ", 1) for x in lines(a)[1:])
    lb = dict(x.split(" = ", 1) for x in lines(b)[1:])
    return {k: (la.get(k), lb.get(k)) for k in sorted(set(la) | set(lb)) if la.get(k) != lb.get(k)}


def write_robot(
    path: str | Path,
    profile_path: str | Path = DEFAULT_PROFILE,
    *,
    weights: str | Path | None = None,
    observe_only: bool = False,
) -> Path:
    """The robot's config file (+ `<file>.provenance.json`): each key with a comment line saying where it is from."""
    cfg, guard, prov = robot_config(profile_path, weights=weights, observe_only=observe_only)
    sim = core_dict()
    diff = differences(cfg, sim)
    for k, (_, simv) in diff.items():
        if k in prov:
            prov[k]["sim_value"] = simv
    p = Path(path)
    meta = prov["_profile"]
    out = [
        "# r1rt / r1guard configuration for the ROBOT (python -m r1mc3.rtconfig --profile robot; read by "
        "rt/src/proc_config.cpp)",
        f"# from {meta['path']} (profile {meta['name']} version {meta['version']}, sha256 {meta['sha256']})",
        f"# provenance of every key: {p.name}.provenance.json; differs from the simulator's file in: "
        + (", ".join(diff) if diff else "nothing"),
    ]
    for line in lines(cfg, guard):
        key = line.split(" = ", 1)[0]
        if key != "format":
            pv = prov.get(key, {})
            where = pv.get("path") or pv.get("from", "?")
            extra = f" ({pv['status']}, {pv['date']})" if "status" in pv else ""
            out.append(f"# {key}: {where}{extra}")
        out.append(line)
    p.write_text("\n".join(out) + "\n")
    Path(f"{p}.provenance.json").write_text(json.dumps(prov, indent=1, sort_keys=True) + "\n")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m r1mc3.rtconfig", description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="the config file to write")
    ap.add_argument("--writer-hz", type=int, default=500)
    ap.add_argument("--policy-hz", type=int, default=50)
    ap.add_argument("--weights", default=None, help="R1W1 weights (default: converted from the repo's policy.onnx)")
    ap.add_argument("--no-guard-gate", action="store_true", help="takeover without a live r1guard (tests only)")
    ap.add_argument(
        "--require-rt", action="store_true", help="refuse TAKEOVER / ON_STAND without real-time grants (the robot)"
    )
    ap.add_argument(
        "--observe-only", action="store_true", help="r1guard never takes the token (guard.takeover_enabled 0)"
    )
    ap.add_argument(
        "--profile",
        choices=("sim", "robot"),
        default="sim",
        help="sim (default): the simulator's numbers (the tests' file); robot: from the robot profile, with provenance",
    )
    ap.add_argument("--profile-path", type=Path, default=DEFAULT_PROFILE, help="the robot profile (--profile robot)")
    a = ap.parse_args(argv)
    if a.profile == "robot":
        sim_only = [
            o
            for o, bad in (
                ("--writer-hz", a.writer_hz != 500),
                ("--policy-hz", a.policy_hz != 50),
                ("--no-guard-gate", a.no_guard_gate),
            )
            if bad
        ]
        if sim_only:
            ap.error(f"{', '.join(sim_only)}: not with --profile robot (the robot's values come from the profile)")
        p = write_robot(a.out, a.profile_path, weights=a.weights, observe_only=a.observe_only)
        print(p)
        return 0
    cfg = core_dict(
        writer_hz=a.writer_hz,
        policy_hz=a.policy_hz,
        require_guard=not a.no_guard_gate,
        require_rt=a.require_rt,
        weights=a.weights,
    )
    p = write(a.out, cfg, {"takeover_enabled": 0} if a.observe_only else None)
    print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
