"""r1hands minimal (session 5, DESIGN.md §2.1, §6): the two Dex3-1 hands' low level over DDS.

`r1mc3.hands.dex3.Dex3Hands(domain)` - per-motor q / dq / kp / kd / tau commands at 100 Hz on rt/dex3/{left,right}/cmd,
the hands' state (joints, temperatures, the 9 x 12 pressure taxels raw) from rt/dex3/{left,right}/state. The grip state
machine, tactile contact and the contact bitmask into the core's segment are later (M6).
"""
