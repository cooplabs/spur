"""The planner's timing check runs and reports every kind (`python -m r1mc3.plan.bench`; HW2's command on the Orin)."""

from __future__ import annotations

from r1mc3.plan import bench


def test_bench_reports_each_kind_and_refusals_come_out_as_expected() -> None:
    res = bench.run(n=4, seed=2)
    k = res["kinds"]
    assert {"hover", "reach", "far", "inside", "styled"} <= set(k)  # styled: session 12's approach_style hovers
    assert k["far"]["as_expected"] == k["far"]["n"] and k["far"]["codes"] == {"UNREACHABLE": k["far"]["n"]}
    assert k["inside"]["as_expected"] == k["inside"]["n"] and set(k["inside"]["codes"]) == {"COLLISION"}
    assert k["reach"]["planned"] >= 3
    for r in k.values():
        assert r["plan_s_p50"] <= r["plan_s_max"]
