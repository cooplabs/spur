# The grasp and the "holding" verdict (`r1mc3.hands.grip`)

[`../DESIGN.md`](../DESIGN.md) §7. Code: `python/r1mc3/hands/grip.py` (the grasp runner, the hand's kinematics,
presets), `python/r1mc3/hands/contact.py` (the grasp's stops into the core's hand-contact record); tests
`tests/unit/test_grip*.py`.

## Why evidence, not force

Our previous controller decided "holding" on the grip force. An EMPTY wrap pressing a flat board read 5.2-5.5 N -
as much as a real hold of a small object (5.2 N): the thumb stalled on the curled index finger and the fingertips
pressed the board. Force alone cannot tell those apart. r1mc3 keeps the closing sequence that held real objects and
replaces the verdict by a table of evidence.

## The interface

```python
g = GraspRunner(hands, "right", grip="wrap", object_class="rigid", width_m=0.048, contact_height=heights)
g.start(now_ns)                      # pre-shape -> close
g.tick(now_ns)                       # r1hands' 100 Hz tick -> "CLOSING" | "CONTACT" | "TENTATIVE" | ...
g.notify_lift(dz_m)                  # the arm layer: the hand's rise since the verdict
g.verdict()                          # GripVerdict(state, reason, evidence, confidence)
g.release(now_ns)                    # open
```

Grips: `wrap` (closes toward the fist, squeeze 0.3 rad past contact), `pinch` (thumb + index tips, 0.4), `tip` (the
same close, 0.25 - for thin layers). Object classes are generic: `rigid` (default), `thin_flexible`, `soft`.
`r1d` exposes it as `hand.grasp` ([`../API.md`](../API.md)).

## The sequence

Pre-shape -> close at 1.2 rad/s -> each finger group stops on its own evidence: a pad rising above its reading at the
close's start for 30 ms, a stall, a creep far behind ("blocked"), a calibrated torque with the joints still, or
"closed" (at the target: on nothing); a stop on the hand itself (the thumb's capsules touching a finger's, from the
kinematics) is "self" -> the squeeze ramps in -> a settle -> the verdict. Heat: the squeeze halves from 60 C, the
grasp ends (relaxed, still powered) at 72 C (on our robot a 0.9 N m hold heats ~6 C/s).

## The evidence (every term reported; the reason = the first failing one)

| Term | rigid / soft | thin_flexible |
|---|---|---|
| opposition / tip pressure | the thumb AND a finger stopped on evidence that is not the hand touching itself | the thumb's tip pad and the finger's tip pad both pressed (the real pads feel only firm fingertip contact) |
| gap / layer | the aperture >= 0.5 x the width (soft 0.3), or >= 6 mm without a width; for a pinch: >= max(6 mm, 0.5 x width - 13 mm) and the tips not met | the clearance between the two tip shells minus the hand's air-pinch reference in [0.5, 12] mm |
| contact height | tips all at the support surface need the aperture to match the width (a low grasp closed on the object's sides), else "pressing the surface" | reported |
| force | grip force >= 1.5 N (necessary, no longer sufficient) | - |
| lift (>= 1 cm) | contact persists and no group runs on > 0.1 rad (soft 0.25) | tip pressure persists and the layer stays >= 0.5 mm |

States: `OPEN -> CLOSING -> CONTACT -> TENTATIVE` (the hand's evidence holds) `-> HOLDING` (after the lift) `|
NOT_HOLDING`. Confidence: the fraction of judged terms that pass, scaled by whether the lift has been seen.

**Contact decides.** After HOLDING, fingers closing further while a contact persists is not a slip - only closing on
nothing is. In the lift window the object has left only when no touching finger group keeps any evidence. A
NOT_HOLDING verdict never relaxes the fingers by itself: the hand keeps its grip until it is told to open.

**thin_flexible needs the hand's air-pinch reference** (`hand.calibrate {kind: air_pinch}`: a tip pinch on nothing,
once per hand): without it the layer term could pass a pinch of air whenever the real tips stop short of the model's
fingertip shapes.

## How it was checked

- In the simulator: the empty-hand false hold of the previous controller comes out NOT_HOLDING; a real small-object
  pinch HOLDING.
- On our robot: scripted grasps of household objects on the stand and standing at a table; the verdict was compared
  with what a person saw and with replays of the recorded grasp traces (`python -m r1mc3.hands.trace replay FILE`):
  the pinch rule above came from those replays (agreement 32 / 33 on one day's traces).

Not solid: the rules rest on one robot's hands and a few dozen real grasps per grip; soft objects that compress under
the squeeze are judged mostly by the lift check.
