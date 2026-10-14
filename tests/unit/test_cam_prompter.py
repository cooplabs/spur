"""The timed prompts for a camera recording (session 12, `r1mc3.cam.prompter`): the plan's own timeline (the schedule
`r1 record run charuco` records against: 81 holds of 2 s after 2 s moves), each prompt at its time, the live count
beside it when r1d answers. New module (absent on the base 1cf3afd)."""

from __future__ import annotations

from r1mc3.cam import prompter


def test_the_charuco_prompts_follow_the_plans_timeline() -> None:
    sched = prompter.schedule("charuco", prompter.parse_params(["distances_m=0.3,0.45,0.6"]))
    assert len(sched) == 81 and sched[0]["t_start"] == 2.0 and sched[-1]["t_end"] == 324.0
    now = [0.0]

    def clock() -> float:
        return now[0]

    def sleep(s: float) -> None:
        now[0] += max(s, 0.01)

    lines: list[str] = []
    n = prompter.run(sched[:3], out=lines.append, count=lambda: "[live: hold still]", clock=clock, sleep=sleep)
    assert n == 3
    assert lines[0].startswith("t    2.0 s  [1/3]  0.30 m, top-left, flat: hold still") and "[live" in lines[0]
    assert lines[1].startswith("t    4.0 s  move -> 0.30 m, top-left, turned")
    assert abs(now[0] - sched[2]["t_end"]) < 0.05  # the prompts kept the plan's times


def test_params_parse_like_the_cli_words() -> None:
    p = prompter.parse_params(["distances_m=0.3,0.6", "hold_s=3.0", "depth_every=5", "label=board"])
    assert p == {"distances_m": [0.3, 0.6], "hold_s": 3.0, "depth_every": 5, "label": "board"}
    assert len(prompter.schedule("charuco", p)) == 2 * 27
