# Checking the backpack's root file system

**Why it matters:** the controller's real-time processes never write files, but the logs are the evidence after
anything goes wrong - they must be trustworthy. Our backpack's root file system (ext4 on the NVMe) had mounted "with
errors" for months (16 ext4 bitmap errors, the first months before we had the robot, never checked in ~240 mounts), and
a telemetry file of the previous controller ended in NUL bytes after a power-off. Check yours once, read-only first:

```bash
findmnt /; lsblk -f; df -h
sudo dumpe2fs -h /dev/nvme0n1p1 2>/dev/null | grep -iE 'state|errors|error count|first error|last error|mount count|last checked'
journalctl -k -b --no-pager | grep -i 'EXT4-fs'
```

(`dumpe2fs -h` reads the superblock: safe while mounted.) `state: clean` and no error count = nothing to do.

## Hard rules

- **Never run e2fsck on a mounted file system** - not on the running root, not even "just to look" with `-n` (on a
  mounted file system it reports phantom errors, and a repair there corrupts it). The check runs only on an UNMOUNTED
  partition, or on a root remounted read-only.
- **Logs first.** Copy everything you have not copied off the backpack (`/var/lib/r1mc3/{logs,records,sessions}`, your
  own data, the site configs) to another machine, with checksums on both sides, before anything is checked.
- The robot on its stand, nothing of ours running, the charger connected: a power loss DURING a repair is the one way
  this makes things worse.

## What we found on L4T R35.3.1 (why the easy way does not work there)

| Usual way | On our backpack |
|---|---|
| a boot-time check (`tune2fs -c 1`, then reboot) | **impossible**: the kernel mounts the root read-write directly (`root=... rw`), the initrd has no fsck, and `systemd-fsck-root` is skipped every boot |
| emergency / single-user mode with the root remounted read-only (`mount -o remount,ro /`, `e2fsck -f /dev/nvme0n1p1`, `reboot -f`) | needs a **local console**: the kernel console is serial only (`console=ttyTCU0 console=ttyAMA0`), the HDMI monitor shows only the desktop (a text VT stays black) - so emergency mode would have no screen and no ssh. Use a USB-serial cable on the debug port |
| boot a root copy from USB, check the unmounted NVMe | the L4T boot menu supports a second `extlinux.conf` entry with `root=PARTUUID=<usb>` (the kernel and initrd still from the NVMe); copy the root to a USB SSD, pick the entry once, `e2fsck -f /dev/nvme0n1p1`, reboot to the default entry. **Never leave a bootable USB drive plugged in** (L4T may prefer it) |

We have not run the check on our robot yet (said plainly): until then every session's logs are copied off and
verified with sha256 (`r1 shutdown` writes the checksums; compare them after the copy).

## Verify (after a check)

```bash
sudo dumpe2fs -h /dev/nvme0n1p1 2>/dev/null | grep -iE 'state|error count|last checked'   # clean, no count, today
ls -la /lost+found                                                                        # files e2fsck could not place
journalctl -k -b | grep -i 'EXT4-fs'                                                      # no "with errors" at this mount
```

If errors come back after a clean check, the NVMe's own health (`sudo nvme smart-log /dev/nvme0`, nvme-cli) is the next
read-only step.
