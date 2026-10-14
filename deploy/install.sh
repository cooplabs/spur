#!/usr/bin/env bash
# Backpack side: install a release side by side with the others (DESIGN.md §11.2). Does NOT switch to it unless
# --activate, never starts / stops / restarts a service, never touches v2 (/home/unitree/r1-mc2).
#   sudo bash deploy/install.sh --tarball r1mc3-<id>.tar.gz [--activate] [--no-build] [--root DIR] [--dry-run]
#        [--rt-budget sysctl|slice|none] [--console-printk "A B C D"|keep] [--no-site] [--rt-dds-rcvbuf bounded|shared]
# Steps: (1) the tarball's sha256 against <tarball>.sha256; (2) unpack into <root>/opt/r1mc3/releases/.tmp-<id>,
# check every file against its MANIFEST, rename to releases/<id> (an existing release is left as it is); (3) build in
# the release (venv with the `dds` extra, the C++ core + guard with CycloneDDS, Release) and link bin/ -> the
# binaries - real root only (--no-build skips it: tests, rehearsals); (4) the site config /etc/r1mc3 from the release's
# templates ONLY where a file is missing; (4c) session 8: the backpack's site settings (deploy/rt/site.sh apply: the
# sysctl.d file with rmem_max 8 MB + the RT budget - HW1 typed them after every power-up - fixes #3 / #4 / #24;
# --rt-budget sysctl (default = the user's decision, 11 Oct 2026) | slice | none, --console-printk (default "1 4 1 7" =
# the user's decision, 11 Oct 2026; keep = no change),
# --no-site skips it); (4d) session 9: the real-time units' CycloneDDS overlay /etc/r1mc3/cyclonedds_rt.xml
# (--rt-dds-rcvbuf bounded = the default, the user's decision 11 Oct 2026: a 256 kB socket receive buffer for r1rt /
# r1guard - #29's open race; shared = no override, the shared file's 4 MB); (5) with --activate: deploy/activate.sh
# (refuses while we own the motors).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/lib.sh"
r1_parse_common "$@"
set -- ${R1_ARGS[@]+"${R1_ARGS[@]}"}
tarball=""; activate=0; build=1; site=1; rt_budget="sysctl"; printk="1 4 1 7"   # the user's decisions, 11 Oct 2026
rt_rcvbuf="bounded"   # session 9: the user's decision, 11 Oct 2026 ("(e) smaller buffer")
while [ $# -gt 0 ]; do
  case "$1" in
    --tarball) tarball="$2"; shift 2 ;;
    --activate) activate=1; shift ;;
    --no-build) build=0; shift ;;
    --no-site) site=0; shift ;;
    --rt-budget) rt_budget="$2"; shift 2 ;;
    --console-printk) printk="$2"; shift 2 ;;
    --rt-dds-rcvbuf) rt_rcvbuf="$2"; shift 2 ;;
    *) r1_die "unknown option $1" ;;
  esac
done
[ -n "$tarball" ] && [ -f "$tarball" ] || r1_die "--tarball FILE is required (made by deploy/release.sh)"
case "$rt_rcvbuf" in bounded|shared) ;; *) r1_die "--rt-dds-rcvbuf wants bounded or shared (got $rt_rcvbuf)" ;; esac
sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }

# (1) integrity of the tarball
[ -f "$tarball.sha256" ] || r1_die "missing $tarball.sha256"
want="$(cut -d' ' -f1 < "$tarball.sha256")"
got="$(sha256 "$tarball" | cut -d' ' -f1)"
[ "$want" = "$got" ] || r1_die "sha256 mismatch: $got != $want"
# the first entry's top directory; `sed -n 1p` reads to the end (`head` would close the pipe early: under pipefail tar
# then dies of SIGPIPE and the install aborts - GNU tar on the backpack, session 6's Linux build check)
top="$(tar -tzf "$tarball" | sed -n 1p | cut -d/ -f1)"
case "$top" in r1mc3-*) ;; *) r1_die "unexpected tarball layout (top directory $top)" ;; esac
id="${top#r1mc3-}"
printf '%s' "$id" | grep -Eq '^[A-Za-z0-9._+-]+$' || r1_die "bad release id $id"

prefix="$(r1_path "$R1_PREFIX")"
rel="$prefix/releases/$id"
if [ -d "$rel" ]; then
  r1_log "release $id is already installed ($rel): left as it is"
else
  # (2) unpack + verify, then rename into place (a half-unpacked release never appears under its id)
  r1_run mkdir -p "$prefix/releases"
  tmp="$prefix/releases/.tmp-$id"
  if [ "$R1_DRY_RUN" = 1 ]; then
    r1_log "dry-run: unpack $tarball -> $rel (MANIFEST checked)"
  else
    rm -rf "$tmp"; mkdir -p "$tmp"
    tar --no-same-owner -xzf "$tarball" -C "$tmp"   # the files belong to root, not to the tarball's owner (fix #6)
    sed -n '/^files:$/,$p' "$tmp/$top/MANIFEST" | tail -n +2 > "$tmp/files.sha256"
    (cd "$tmp/$top" && sha256 -c "$tmp/files.sha256" >/dev/null 2>&1) || r1_die "MANIFEST check failed for $id"
    rm -f "$tmp/files.sha256"
    mv "$tmp/$top" "$rel"; rmdir "$tmp"
    r1_log "installed release $id -> $rel"
  fi
fi

# (3) build on the backpack (real root only). bin/ is the stable entry the units use.
# Session 6 (the Linux build check, Ubuntu 20.04 arm64 = the Orin's L4T 35.3.1): the backpack's system cmake (3.16)
# is older than rt/CMakeLists' 3.24 and its python3 (3.8) older than 3.11, and a --no-dev venv has no nanobind / onnx
# - so the build uses the release venv's own tools (the `build` dependency group: cmake, ninja, nanobind, onnx) and
# Python 3.11, and builds into build/rt: r1rt / r1guard (apps/), the r1rt_py module r1d / r1log load
# (r1mc3.log._native: <release>/build/rt) and the converted policy weights rtconfig finds there (policy/).
# Session 12 (HW2 day 1, 12 Oct 2026): uv's managed Python 3.11 goes to <prefix>/python (/opt/r1mc3/python on the
# robot), never uv's default under the invoking user's home - under sudo that is /root/.local/share/uv/python, which the
# units' user cannot read (the venv's interpreter is a link into it: r1d / r1log would not start). HW1 and HW2 day 1
# typed `sudo env UV_PYTHON_INSTALL_DIR=/opt/r1mc3/python bash .../install.sh ...` by hand (docs/sessions/HW1.md P2);
# an explicit UV_PYTHON_INSTALL_DIR from the caller still wins.
if [ "$build" = 1 ]; then
  venv="$rel/venv"
  dds_prefix="${CYCLONEDDS_HOME:-/opt/r1mc3/cyclonedds-0.10.2}"
  py_dir="${UV_PYTHON_INSTALL_DIR:-$prefix/python}"
  r1_sys env UV_PROJECT_ENVIRONMENT="$venv" UV_PYTHON_INSTALL_DIR="$py_dir" CYCLONEDDS_HOME="$dds_prefix" \
    uv sync --project "$rel" --frozen --no-dev --group build --extra dds --extra perception
  r1_sys "$venv/bin/cmake" -S "$rel/rt" -B "$rel/build/rt" -G Ninja -DCMAKE_MAKE_PROGRAM="$venv/bin/ninja" \
    -DCMAKE_BUILD_TYPE=Release -DR1RT_WITH_CYCLONEDDS=ON -DR1RT_BUILD_TESTS=OFF -DCMAKE_PREFIX_PATH="$dds_prefix" \
    -DPython_EXECUTABLE="$venv/bin/python" -DPython3_EXECUTABLE="$venv/bin/python"
  r1_sys "$venv/bin/cmake" --build "$rel/build/rt"
else
  r1_log "build skipped (--no-build)"
fi
if [ "$R1_DRY_RUN" = 0 ] && [ -d "$rel" ]; then
  mkdir -p "$rel/bin"
  ln -sfn ../build/rt/apps/r1rt "$rel/bin/r1rt"
  ln -sfn ../build/rt/apps/r1guard "$rel/bin/r1guard"
  # bin/python: a wrapper, NOT a symlink - Python finds its venv by the path it is started with (venv/pyvenv.cfg next
  # to venv/bin/); started as /opt/r1mc3/current/bin/python through a symlink it ran without the venv and r1d / r1log
  # could not import r1mc3 (session 6, the Linux build check: install.sh run for real in an Ubuntu 20.04 container)
  rm -f "$rel/bin/python"
  printf '%s\n' '#!/bin/sh' '# the release venv'"'"'s interpreter, started by its own path (deploy/install.sh)' \
    'exec "$(dirname "$0")/../venv/bin/python" "$@"' > "$rel/bin/python"
  chmod 0755 "$rel/bin/python"
fi

# (4) site config: only what is missing (never overwrite the operator's files)
etc="$(r1_path "$R1_ETC")"
r1_run mkdir -p "$etc"
if [ -f "$etc/cyclonedds.xml" ]; then
  r1_log "kept $etc/cyclonedds.xml"
else
  r1_run cp "$here/config/cyclonedds.xml" "$etc/cyclonedds.xml"
fi
[ -f "$etc/r1rt.cfg" ] || r1_log "NOTE: $etc/r1rt.cfg is missing: write it with python -m r1mc3.rtconfig --profile robot before the first start"

# (4d) session 9 (#29's open race): the real-time units' CycloneDDS overlay. r1rt / r1rt-stand / r1guard /
# r1guard-observe read CYCLONEDDS_URI=file:///etc/r1mc3/cyclonedds.xml,file:///etc/r1mc3/cyclonedds_rt.xml (a later
# setting overrides an earlier one): `bounded` (default) = config/cyclonedds_rt.xml, a 256 kB socket receive buffer
# (the backlog a stop leaves to drain before a live lowstate: ~0.2 s instead of ~1.6 s+); `shared` =
# config/cyclonedds_rt_shared.xml, no override (the shared 4 MB, as before). This file is the release's, not the
# operator's: it is (re)written from the option every install; a different existing file is kept as .bak-<UTC> first.
rt_dds_src="$here/config/cyclonedds_rt.xml"; [ "$rt_rcvbuf" = shared ] && rt_dds_src="$here/config/cyclonedds_rt_shared.xml"
rt_dds="$etc/cyclonedds_rt.xml"
if [ -f "$rt_dds" ] && cmp -s "$rt_dds" "$rt_dds_src"; then
  r1_log "kept $rt_dds (--rt-dds-rcvbuf $rt_rcvbuf)"
else
  if [ -f "$rt_dds" ]; then r1_run cp "$rt_dds" "$rt_dds.bak-$(date -u +%Y%m%dT%H%M%SZ)"; fi
  r1_run cp "$rt_dds_src" "$rt_dds"
  r1_log "wrote $rt_dds (--rt-dds-rcvbuf $rt_rcvbuf; effective at the next start of r1rt / r1guard)"
fi

# (4b) the services' data folders the units name (session 6): logs/ = r1log's --out = r1d's R1LOG_DIR, records/ =
# R1_RECORD_DIR, sessions/ = R1_SESSION_DIR (`r1 shutdown`'s log save), owned by the units' user. systemd's
# StateDirectory= in r1d / r1log creates and re-owns them at start too; made here so a manual start finds them.
state="$(r1_path "$R1_STATE")"
for d in $R1_STATE_SUBDIRS; do r1_run mkdir -p "$state/$d"; done
for d in "" $R1_STATE_SUBDIRS; do r1_sys chown "$R1_USER:$R1_USER" "$state${d:+/$d}"; done

# (4c) the site settings (deploy/rt/site.sh; the installed release's own copy runs the same code). Other r1mc3 site
# settings already in place are kept (site.sh refuses to overwrite them: `site.sh undo` first, the operator's call).
if [ "$site" = 1 ]; then
  sargs=(apply --root "$R1_ROOT" --rt-budget "$rt_budget" --console-printk "$printk"); [ "$R1_DRY_RUN" = 1 ] && sargs+=(--dry-run)
  bash "$here/rt/site.sh" "${sargs[@]}" || r1_log "site settings NOT changed (see above; deploy/rt/site.sh show)"
else
  r1_log "site settings skipped (--no-site)"
fi

# (5) optional switch
if [ "$activate" = 1 ] && [ "$R1_DRY_RUN" = 1 ] && [ ! -d "$rel" ]; then
  r1_log "dry-run: would activate $id (deploy/activate.sh: refused while we own the motors)"
elif [ "$activate" = 1 ]; then
  args=(--root "$R1_ROOT" --id "$id"); [ "$R1_DRY_RUN" = 1 ] && args+=(--dry-run)
  bash "$here/activate.sh" "${args[@]}"
else
  r1_log "not activated: bash deploy/activate.sh --id $id (current: $(r1_current_id))"
fi
