# mypy: disable-error-code="type-arg,arg-type"
# (CycloneDDS IDL field syntax: types.sequence[T] is not a mypy generic; the module is generated)
"""unitree_api Request_ / Response_ (the envelope of Unitree's RPC services: motion switcher, loco, ...) for
CycloneDDS's Python binding.

Generated from OUR reconstructed IDL (`rt/idl/src/unitree/idl/api/*.idl`, proven byte-identical to Unitree's
generated C++ headers incl. the XTypes type blobs - rt/idl/src/unitree/idl/api/SOURCE.md) by CycloneDDS 0.10.2's idlc
Python backend (10 Oct 2026, v3 session 5) and merged into one module (forward references became class references).
"""

# No `from __future__ import annotations`: CycloneDDS's IDL reads the field annotations at run time.
from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types


@dataclass
@annotate.final
@annotate.autoid("sequential")
class RequestIdentity_(idl.IdlStruct, typename="unitree_api.msg.dds_.RequestIdentity_"):
    id: types.int64
    api_id: types.int64


@dataclass
@annotate.final
@annotate.autoid("sequential")
class RequestLease_(idl.IdlStruct, typename="unitree_api.msg.dds_.RequestLease_"):
    id: types.int64


@dataclass
@annotate.final
@annotate.autoid("sequential")
class RequestPolicy_(idl.IdlStruct, typename="unitree_api.msg.dds_.RequestPolicy_"):
    priority: types.int32
    noreply: bool


@dataclass
@annotate.final
@annotate.autoid("sequential")
class RequestHeader_(idl.IdlStruct, typename="unitree_api.msg.dds_.RequestHeader_"):
    identity: RequestIdentity_
    lease: RequestLease_
    policy: RequestPolicy_


@dataclass
@annotate.final
@annotate.autoid("sequential")
class Request_(idl.IdlStruct, typename="unitree_api.msg.dds_.Request_"):
    header: RequestHeader_
    parameter: str
    binary: types.sequence[types.uint8]


@dataclass
@annotate.final
@annotate.autoid("sequential")
class ResponseStatus_(idl.IdlStruct, typename="unitree_api.msg.dds_.ResponseStatus_"):
    code: types.int32


@dataclass
@annotate.final
@annotate.autoid("sequential")
class ResponseHeader_(idl.IdlStruct, typename="unitree_api.msg.dds_.ResponseHeader_"):
    identity: RequestIdentity_
    status: ResponseStatus_


@dataclass
@annotate.final
@annotate.autoid("sequential")
class Response_(idl.IdlStruct, typename="unitree_api.msg.dds_.Response_"):
    header: ResponseHeader_
    data: str
    binary: types.sequence[types.uint8]


def request(req_id: int, api_id: int, parameter: str = "", *, lease: int = 0, noreply: bool = False) -> Request_:
    return Request_(
        header=RequestHeader_(
            identity=RequestIdentity_(id=req_id, api_id=api_id),
            lease=RequestLease_(id=lease),
            policy=RequestPolicy_(priority=0, noreply=noreply),
        ),
        parameter=parameter,
        binary=[],
    )


def response(identity: RequestIdentity_, code: int, data: str = "") -> Response_:
    return Response_(
        header=ResponseHeader_(
            identity=RequestIdentity_(id=identity.id, api_id=identity.api_id), status=ResponseStatus_(code=code)
        ),
        data=data,
        binary=[],
    )
