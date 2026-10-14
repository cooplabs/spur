# unitree_api request / response types (the RPC envelope of Unitree's services) - reconstructed IDL

Unitree's SDK ships only the CycloneDDS C++ code generated from these types (namespace `unitree_api::msg::dds_`),
not the `.idl` files. The eight `.idl` files here are reconstructed from that generated code and **proven**:
CycloneDDS 0.10.2's idlc (`idlc -x final -Wno-implicit-extensibility -l libcycloneddsidlcxx`, our build
`~/.cache/r1mc3/cyclonedds-0.10.2`, the same command rt/io.cmake uses for the hg types) regenerates all eight headers
byte for byte from line 10 on, ignoring the include-guard and `#include` lines - including the XTypes type-map /
type-info blobs that DDS peers match (checked 10 Oct 2026, v3 session 5, the bridge agent; the comparison command is
in docs/sim/DDS_BRIDGE.md). `sequence<octet>` = `std::vector<uint8_t>` with TK_BYTE (CycloneDDS's Python binding maps
`types.uint8` to TK_BYTE too). Used by `python/r1mc3/dds/api_types.py` (idlc's Python backend, merged into one
module) and the motion-switcher RPC (`python/r1mc3/dds/rpc.py`).

Fetched single raw files (read, never compiled or run; byte-identical copies in `rt/idl/unitree_api/`, checked by tests/unit/test_dds_rpc.py), unitree_sdk2 @ 63096d0ac0c5d2dec9d6e0c22cd5233410ca2f36
(BSD 3-Clause, rt/idl/LICENSE.unitree_sdk2), `raw.githubusercontent.com/unitreerobotics/unitree_sdk2/<commit>/`:

| File | sha256 |
|---|---|
| `include/unitree/robot/internal/internal_idl_decl/Request_.hpp` | `889137406890f4d91cc3061f84aa3e54fececfa52eda14c7f14262677e2b36db` |
| `.../RequestHeader_.hpp` | `2d46dfeaa302281bccf06578959b34ca239d6e3334513d8fdd36677133488e4b` |
| `.../RequestIdentity_.hpp` | `584c0b8cc972f7787a4b8149dbfd72b566d353a14fd6e4d4fe1d586fd20b18a9` |
| `.../RequestLease_.hpp` | `46eb42c95bad732e9d31ece7060b23c6f0bb3f4d4720cc7209af3e9a61c9cca9` |
| `.../RequestPolicy_.hpp` | `af2ba2ef86f44f5be4cf665cf757e34c66cd1df27399295b3b264769a9024da4` |
| `.../Response_.hpp` | `f767bb234ce49ab9406abf23feca939737fb9128a67aa6e1ff19a6f0acef8fab` |
| `.../ResponseHeader_.hpp` | `8dddb858a74e9916c708cca216643dfaf47e01b62ffa437084293889ccdb6317` |
| `.../ResponseStatus_.hpp` | `60d97a503802ee87f35521ba2429369f229b4ea66eab9ce350d442ba4a4aa601` |
| `include/unitree/robot/b2/motion_switcher/motion_switcher_api.hpp` (service name `motion_switcher`, api version 1.0.0.1, ids CheckMode 1001 / SelectMode 1002 / ReleaseMode 1003 / SetSilent 1004 / GetSilent 1005, the JSON `{"name", "form"}`) | `50d1ddd901bb72ae354cce9cdaae5c1173c06d03d3c59c0baf0d7bd2a9ee3366` |
| `include/unitree/robot/b2/motion_switcher/motion_switcher_client.hpp` (CheckMode(form, name), SelectMode(nameOrAlias), ReleaseMode(), SetSilent, GetSilent) | `84336c912c9776afc2314b74f9e67ff10a1856747f2914289f89f3c07c942407` |
| `include/unitree/robot/b2/motion_switcher/motion_switcher_error.hpp` (codes 7001-7009) | `e411b1ca530d55b2fc3e41ab264a067f0c767bd63181f428a5b202750b4f481c` |

Not verified against the robot (no robot this session): the request / response topic names
`rt/api/motion_switcher/request|response` (the SDK's client channel naming, as v2's unitree_sdk2py client used it on
the R1) and the CheckMode payload (`{"form":"0","name":"ai"}` while the built-in runs, `name` "" after a release: v2
runtime.py's reading) - HW1's read-only discovery checks both.
