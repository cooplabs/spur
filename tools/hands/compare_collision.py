"""Compare the Dex3-1 collision primitives (r1sim.dex3.hand_xml) with Unitree's link meshes (models/dex3_1/meshes).

Session 3 (the user's note "the hands are not super accurate"; the A9 short-grasp cause decomposition asks whether the
hand geometry explains part of the 1.5-3 cm shortfall). Numbers, all in the LINK frame (or the palm frame for the
presets), millimetres:

* per link: the mesh's and the primitives' extents along the link's distal axis (length), the closing axis
  (thickness: the grasp side is +g y for the fingers / palm and +x for the thumb links, g = +1 right, -1 left) and
  the third axis (width);
* mesh outside the primitives: the signed distance of every mesh vertex to the union of the link's collision geoms
  (analytic SDFs; > 0 = the real shape sticks out of the simulated one);
* primitives outside the mesh: the primitives' surface samples that lie outside the mesh's convex hull (MuJoCo's
  hull of the link mesh; the CAD meshes are hollow shells, so the hull is the outer envelope that contact sees) and
  by how much (> 0 = the simulated shape sticks out of the real one);
* the pads: where each pad box's outer face sits vs the mesh surface along the pad normal (a ray at the pad centre);
* fingertips: the mesh's distal extreme vs the capsule end vs the tip site (dex3.TIP), and their positions in the
  palm frame for the open / relax / fist / pinch presets; v2's open-hand grasp centre (0.5 thumb tip + 0.25 (index +
  middle tips), r1mc2 api.py:1724-1728) from the tip sites vs from the mesh tips.

Usage: uv run python tools/hands/compare_collision.py [--side right] [--json out.json] [--md out.md]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sim"))
from r1sim import dex3, model  # noqa: E402

FloatArray = npt.NDArray[np.float64]
MM = 1000.0


def stl_mesh(path: Path) -> tuple[FloatArray, FloatArray]:
    """(unique vertices (n, 3), triangles (m, 3, 3)) of a binary STL, metres, link frame."""
    b = path.read_bytes()
    n = struct.unpack("<I", b[80:84])[0]
    rec = np.frombuffer(b[84 : 84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    tri = rec["v"].astype(np.float64)
    return np.unique(tri.reshape(-1, 3), axis=0), tri


def _mat(quat: FloatArray) -> FloatArray:
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, np.asarray(quat, dtype=np.float64))
    return r.reshape(3, 3)


def primitives(m: mujoco.MjModel, side: str, link: str) -> list[dict[str, Any]]:
    """The link's collision geoms (body frame): type, size, pos, rotation, name, pad?"""
    b = m.body(f"{side}_hand_{link}_link").id
    out = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != b or not m.geom_contype[g]:
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        out.append({"name": name, "type": int(m.geom_type[g]), "size": m.geom_size[g].copy(),
                    "pos": m.geom_pos[g].copy(), "R": _mat(m.geom_quat[g]), "pad": "_pad_" in name})  # fmt: skip
    return out


def sdf(p: FloatArray, g: dict[str, Any]) -> FloatArray:
    """Signed distance of points p (n, 3) (body frame) to one primitive (> 0 outside)."""
    loc = (p - g["pos"]) @ g["R"]
    s = g["size"]
    t = g["type"]
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.asarray(np.linalg.norm(loc, axis=1) - s[0])
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        z = np.clip(loc[:, 2], -s[1], s[1])
        return np.asarray(np.linalg.norm(loc - np.stack([0 * z, 0 * z, z], axis=1), axis=1) - s[0])
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        q = np.abs(loc) - s[:3]
        return np.asarray(np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0))
    raise ValueError(t)


def surface_samples(g: dict[str, Any], n: int = 24) -> tuple[FloatArray, FloatArray]:
    """Points on a primitive's surface and their outward normals (body frame)."""
    s, t = g["size"], g["type"]
    pts: list[FloatArray] = []
    nrm: list[FloatArray] = []
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        for ax in range(3):
            for sg in (-1.0, 1.0):
                u, v = [a for a in range(3) if a != ax]
                for a in np.linspace(-0.9, 0.9, 7):
                    for c in np.linspace(-0.9, 0.9, 7):
                        p = np.zeros(3)
                        p[ax], p[u], p[v] = sg * s[ax], a * s[u], c * s[v]
                        nn = np.zeros(3)
                        nn[ax] = sg
                        pts.append(p)
                        nrm.append(nn)
    elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        for z in np.linspace(-s[1], s[1], 9):
            for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
                nn = np.array([np.cos(a), np.sin(a), 0.0])
                pts.append(nn * s[0] + [0, 0, z])
                nrm.append(nn)
        for sg in (-1.0, 1.0):  # the hemispherical caps
            for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
                for el in np.linspace(0.2, 1.4, 4):
                    nn = np.array([np.cos(a) * np.cos(el), np.sin(a) * np.cos(el), sg * np.sin(el)])
                    pts.append(nn * s[0] + [0, 0, sg * s[1]])
                    nrm.append(nn)
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
        for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
            for el in np.linspace(-1.3, 1.3, 7):
                nn = np.array([np.cos(a) * np.cos(el), np.sin(a) * np.cos(el), np.sin(el)])
                pts.append(nn * s[0])
                nrm.append(nn)
    P, N = np.array(pts), np.array(nrm)
    return P @ g["R"].T + g["pos"], N @ g["R"].T


class MeshRay:
    """mj_ray against one link mesh's triangles (a one-geom model, mesh in the link frame at the origin)."""

    def __init__(self, path: Path) -> None:
        xml = (
            f'<mujoco><asset><mesh name="m" file="{path}"/></asset>'
            '<worldbody><geom name="m" type="mesh" mesh="m"/></worldbody></mujoco>'
        )
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        self.gid = np.zeros(1, dtype=np.int32)

    def hull_planes(self) -> tuple[FloatArray, FloatArray]:
        """The mesh's convex hull (MuJoCo's, from mesh_graph) as outward planes n . x <= c in the link frame."""
        m = self.m
        a = int(m.mesh_graphadr[0])
        nv, nf = int(m.mesh_graph[a]), int(m.mesh_graph[a + 1])
        # layout (MuJoCo mjModel docs): numvert, numface, vert_edgeadr[nv], vert_globalid[nv], edge_localid[nv+3nf],
        # face_globalid[3nf]
        off = a + 2 + nv + nv + nv + 3 * nf
        faces = np.asarray(m.mesh_graph[off : off + 3 * nf]).reshape(nf, 3)
        va = int(m.mesh_vertadr[0])
        v = np.asarray(m.mesh_vert[va : va + int(m.mesh_vertnum[0])], dtype=np.float64)
        R = _mat(m.geom_quat[0])
        vw = v @ R.T + m.geom_pos[0]  # mesh frame -> link frame
        t = vw[faces]
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        keep = np.linalg.norm(n, axis=1) > 2e-9  # sliver faces (area < 1e-9 m^2) have meaningless normals
        n, t = n[keep], t[keep]
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        c = np.einsum("ij,ij->i", n, t[:, 0])
        centre = vw.mean(axis=0)
        flip = (n @ centre - c) > 0  # outward: the centroid must be inside
        n[flip] *= -1
        c[flip] *= -1
        return n, c

    def hit(self, p: FloatArray, v: FloatArray) -> float:
        """Distance along unit v from p to the mesh surface (-1 = no hit)."""
        return float(mujoco.mj_ray(self.m, self.d, np.asarray(p, float), np.asarray(v, float), None, 1, -1, self.gid))

    def surface_along(self, p: FloatArray, n: FloatArray, reach: float = 0.03) -> float | None:
        """Signed position of the mesh surface along n through p: + = the surface lies OUTSIDE p (along n). Casts
        from p + reach*n back along -n (the first hit is the outermost surface)."""
        d = self.hit(p + reach * n, -n)
        return None if d < 0 else reach - d


def axes(side: str, link: str) -> tuple[int, FloatArray, int, int]:
    """(distal axis index, grasp-side unit vector, thickness axis index, width axis index) in the link frame."""
    g = -1.0 if side == "left" else 1.0
    if link.startswith("thumb"):
        return 1, np.array([1.0, 0.0, 0.0]), 0, 2  # the thumb chain runs along g*y; its pads face +x
    if link == "palm":
        return 0, np.array([0.0, g, 0.0]), 1, 2
    return 0, np.array([0.0, g, 0.0]), 1, 2


def compare(side: str = "right") -> dict[str, Any]:
    m = model.build_model("real")
    g = -1.0 if side == "left" else 1.0
    out: dict[str, Any] = {"side": side, "links": {}, "pads": {}, "tips": {}, "presets": {}}
    tip_mesh_local: dict[str, FloatArray] = {}
    for link in ("palm", *dex3.NAMES):
        path = dex3.mesh_file(side, link)
        verts, _ = stl_mesh(path)
        prims = primitives(m, side, link)
        ray = MeshRay(path)
        dist_ax, grasp, th_ax, w_ax = axes(side, link)
        sgn_d = g if link.startswith("thumb") and dist_ax == 1 else 1.0
        # extents
        pv = np.concatenate([surface_samples(p)[0] for p in prims if not p["pad"]] or [np.zeros((1, 3))])
        allp = np.concatenate([surface_samples(p)[0] for p in prims] or [np.zeros((1, 3))])
        row: dict[str, Any] = {
            "mesh_min_mm": (verts.min(0) * MM).round(1).tolist(),
            "mesh_max_mm": (verts.max(0) * MM).round(1).tolist(),
            "prim_min_mm": (allp.min(0) * MM).round(1).tolist(),
            "prim_max_mm": (allp.max(0) * MM).round(1).tolist(),
            "length_mesh_mm": round(float((verts[:, dist_ax] * sgn_d).max() * MM), 1),
            "length_prim_mm": round(float((pv[:, dist_ax] * sgn_d).max() * MM), 1),
            "thickness_mesh_mm": round(float(np.ptp(verts[:, th_ax]) * MM), 1),
            "thickness_prim_mm": round(float(np.ptp(pv[:, th_ax]) * MM), 1),
            "width_mesh_mm": round(float(np.ptp(verts[:, w_ax]) * MM), 1),
            "width_prim_mm": round(float(np.ptp(pv[:, w_ax]) * MM), 1),
        }
        # grasp-side surface: mesh extreme along the grasp normal vs the primitives' (incl. pads)
        row["grasp_face_mesh_mm"] = round(float((verts @ grasp).max() * MM), 1)
        row["grasp_face_prim_mm"] = round(float((allp @ grasp).max() * MM), 1)
        if prims:
            dmin = np.min(np.stack([sdf(verts, p) for p in prims]), axis=0)
            row["mesh_outside_prims_max_mm"] = round(float(dmin.max() * MM), 1)
            row["mesh_outside_prims_p95_mm"] = round(float(np.percentile(dmin, 95) * MM), 1)
            row["mesh_outside_prims_frac_gt3mm"] = round(float((dmin > 0.003).mean()), 3)
            # primitive surface samples outside the mesh's OUTER ENVELOPE (its convex hull: the CAD meshes are hollow
            # shells, so "inside the material" would count the cavities as outside): max over the hull planes
            # (= the distance for points beyond a face; a lower bound near edges)
            P = np.concatenate([surface_samples(p)[0] for p in prims if not p["pad"]])
            hn, hc = ray.hull_planes()
            dh = (P @ hn.T - hc).max(axis=1)
            row["prims_outside_hull_frac"] = round(float((dh > 0.0005).mean()), 3)
            row["prims_outside_hull_max_mm"] = round(float(dh.max() * MM), 1)
            row["prims_outside_hull_p95_mm"] = round(float(np.percentile(dh, 95) * MM), 1)
        out["links"][link] = row
        # pads: outer face centre vs mesh surface along the pad normal
        for p in prims:
            if not p["pad"]:
                continue
            ax = int(np.argmin(p["size"][:3]))
            nl = np.zeros(3)
            nl[ax] = 1.0
            n = p["R"] @ nl
            if np.dot(n, p["pos"]) < 0 or (link == "palm" and np.dot(n, grasp) < 0):
                n = -n
            face = p["pos"] + n * p["size"][ax]
            s = ray.surface_along(face, n)
            out["pads"][p["name"]] = {
                "link": link,
                "face_mm": (face * MM).round(1).tolist(),
                "normal": n.round(3).tolist(),
                "mesh_surface_beyond_pad_face_mm": None if s is None else round(s * MM, 1),
            }
        # tips
        if link in ("thumb_2", "middle_1", "index_1"):
            k = int(np.argmax(verts[:, dist_ax] * sgn_d))
            tip_mesh_local[link] = verts[k]
            caps = [p for p in prims if p["type"] == mujoco.mjtGeom.mjGEOM_CAPSULE]
            cap_tip = max((sdf_end(c, dist_ax, sgn_d) for c in caps), default=float("nan"))
            out["tips"][link] = {
                "mesh_tip_mm": (verts[k] * MM).round(1).tolist(),
                "mesh_reach_mm": round(float(verts[k, dist_ax] * sgn_d * MM), 1),
                "capsule_reach_mm": round(cap_tip * MM, 1),
                "tip_site_mm": round(dex3.TIP * MM, 1),
            }
    # presets: tips in the palm frame; v2's open grasp centre
    tip_site_local = {
        "thumb_2": np.array([0.0, g * dex3.TIP, 0.0]),
        "middle_1": np.array([dex3.TIP, 0.0, 0.0]),
        "index_1": np.array([dex3.TIP, 0.0, 0.0]),
    }
    for pname in ("open", "relax", "fist", "pinch"):
        poses = dex3.link_poses(side, dex3.preset(pname, side))
        row = {}
        for link in ("thumb_2", "middle_1", "index_1"):
            pos, quat = poses[link]
            R = _mat(quat)
            site = pos + R @ tip_site_local[link]
            mesh = pos + R @ tip_mesh_local[link]
            row[link] = {"site_mm": (site * MM).round(1).tolist(), "mesh_mm": (mesh * MM).round(1).tolist(),
                         "mesh_minus_site_mm": ((mesh - site) * MM).round(1).tolist()}  # fmt: skip
        out["presets"][pname] = row
    op = out["presets"]["open"]
    gc_site = 0.5 * np.array(op["thumb_2"]["site_mm"]) + 0.25 * (
        np.array(op["index_1"]["site_mm"]) + np.array(op["middle_1"]["site_mm"])
    )
    gc_mesh = 0.5 * np.array(op["thumb_2"]["mesh_mm"]) + 0.25 * (
        np.array(op["index_1"]["mesh_mm"]) + np.array(op["middle_1"]["mesh_mm"])
    )
    out["grasp_centre_open"] = {
        "from_tip_sites_mm": gc_site.round(1).tolist(),
        "from_mesh_tips_mm": gc_mesh.round(1).tolist(),
        "mesh_minus_sites_mm": (gc_mesh - gc_site).round(1).tolist(),
    }
    return out


def sdf_end(c: dict[str, Any], dist_ax: int, sgn: float) -> float:
    """The capsule's farthest point along the distal axis (link frame)."""
    s = c["size"]
    ends = [c["pos"] + c["R"] @ np.array([0.0, 0.0, h]) for h in (-s[1], s[1])]
    return float(max(e[dist_ax] * sgn for e in ends) + s[0])


def markdown(r: dict[str, Any]) -> str:
    L = [f"### Dex3-1 collision primitives vs Unitree's meshes ({r['side']} hand; mm, link frame)", ""]
    L.append("| link | length mesh / prim | thickness mesh / prim | width mesh / prim | grasp face mesh / prim "
             "| mesh outside prims max (p95) | prims outside the mesh hull: frac, max (p95) |")  # fmt: skip
    L.append("|---|---|---|---|---|---|---|")
    for k, v in r["links"].items():
        L.append(
            f"| {k} | {v['length_mesh_mm']} / {v['length_prim_mm']} | {v['thickness_mesh_mm']} / "
            f"{v['thickness_prim_mm']} | {v['width_mesh_mm']} / {v['width_prim_mm']} | {v['grasp_face_mesh_mm']} / "
            f"{v['grasp_face_prim_mm']} | {v.get('mesh_outside_prims_max_mm')} ({v.get('mesh_outside_prims_p95_mm')})"
            f" | {v.get('prims_outside_hull_frac')}, {v.get('prims_outside_hull_max_mm')} "
            f"({v.get('prims_outside_hull_p95_mm')}) |"
        )
    L += ["", "| pad | link | mesh surface beyond the pad face (mm, + = the real hand is thicker) |", "|---|---|---|"]
    for k, v in r["pads"].items():
        L.append(f"| {k} | {v['link']} | {v['mesh_surface_beyond_pad_face_mm']} |")
    L += ["", "| tip | mesh reach | capsule reach | tip site |", "|---|---|---|---|"]
    for k, v in r["tips"].items():
        L.append(f"| {k} | {v['mesh_reach_mm']} | {v['capsule_reach_mm']} | {v['tip_site_mm']} |")
    L += ["", "| preset | link | tip site (palm frame) | mesh tip | mesh - site |", "|---|---|---|---|---|"]
    for pn, row in r["presets"].items():
        for k, v in row.items():
            L.append(f"| {pn} | {k} | {v['site_mm']} | {v['mesh_mm']} | {v['mesh_minus_site_mm']} |")
    gc = r["grasp_centre_open"]
    L += ["", f"v2's open-hand grasp centre (palm frame): tip sites {gc['from_tip_sites_mm']}, mesh tips "
          f"{gc['from_mesh_tips_mm']}, difference {gc['mesh_minus_sites_mm']} mm."]  # fmt: skip
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--side", default="right", choices=dex3.SIDES)
    ap.add_argument("--json")
    ap.add_argument("--md")
    a = ap.parse_args(argv)
    r = compare(a.side)
    md = markdown(r)
    print(md)
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1))
    if a.md:
        Path(a.md).write_text(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
