# A1 sim2sim reference data (DESIGN.md §13.1 A1)

Made 10 Oct 2026 (v3 session 2) with `tools/sim2sim/mjlab_reference.py` in the training venv (`~/.venvs/r1-train`:
mjlab 1.2.0, mujoco + mujoco-warp 3.5.0, warp 1.12.0; read-only use) from mjlab's OWN compiled training model of the
live policy R1MC-Manip2-Lean (`policies/r1mc2_manip2_lean_20261006T0421Z`: the robot spec as unitree_rl_mjlab@1425b15
`r1_constants.get_spec` + the training kit's `get_spec_with_hands`, then mjlab's EntityCfg / Scene / MujocoCfg with
params/env.yaml's actuators, collisions, initial state and sim options; no unitree_rl_mjlab code was executed).

| File | sha256 | What |
|---|---|---|
| `mjlab_model_fields.npz` | `b858bff91b92adc1784071f638c7f65360d373011267749eb703ede42e7ded94` | the compiled model's physics fields by name (`r1sim.train_equiv.model_fields`; `tools/sim2sim/model_diff.py` diffs it against ours) |
| `mjlab_cpu_S0.npz` | `355c6df9ac5672f8cb9950fbdba7d6cbfeee09ecf94f331c7006ebda91418e33` | 10 s stand (S0) stepped with MuJoCo CPU 3.5.0 (`mj_step`) on mjlab's model |
| `mjlab_cpu_S1.npz` | `86ba012936fbbf2f5be1e9e7ce2dae0e7eb8156306ca68cb24255814f294f79b` | S1: stand 2 s, vx 0.3 m/s 2-6 s, stand to 10 s, MuJoCo CPU |
| `mjlab_warp_S0.npz` | `7001cb704fd1ccf42ced63805e92882dd7533db8cbc394b7666ff60d6bf50a85` | S0 stepped with MuJoCo Warp 3.5.0 (float32, Warp's CPU device) through mjlab's own `Simulation` (the training engine) |
| `mjlab_warp_S1.npz` | `84b75a556a6c06e0a3eac5d38059da76a18b2c9e7f01e04ce700ac90dbaf90f9` | S1, MuJoCo Warp |

Each rollout: per policy step k (t = 0.02 k, before that step's action): `t`, `q` (24 policy joints, ONNX order),
`base_pos`, `base_quat` (w, x, y, z), `action` (14), `cmd`, `obs_head` (the first 5 observations), `joint_names`.
Start: training's reset (base 0.745 m upright, POSE_STAND, the soles 10.9 mm above the floor); arms held at their
default angles; training's exact action scale and gains (`PolicyInterface.training`); noise, delays and domain
randomisation off. Policy executor: `r1mc3.policy.onnx_np.OnnxModel` (bit-identical to v2's NumPy executor), the
observation code `r1sim.train_equiv.mjlab_observation` - the same on both sides.

Regenerate (and update the checksums above):

```bash
~/.venvs/r1-train/bin/python tools/sim2sim/mjlab_reference.py model --out /tmp/a1/mjlab_r1.mjb
uv run python tools/sim2sim/model_diff.py /tmp/a1/mjlab_r1.mjb --save tests/assets/a1/mjlab_model_fields.npz
for e in cpu warp; do for s in S0 S1; do
  ~/.venvs/r1-train/bin/python tools/sim2sim/mjlab_reference.py rollout --engine $e --script $s \
    --out tests/assets/a1/mjlab_${e}_${s}.npz; done; done
```
