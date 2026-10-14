# ruff: noqa: I001, BLE001
# pylint: disable=broad-exception-caught

from __future__ import annotations

from concurrent import futures
from typing import Any

import grpc

from psrl.utils.logger import get_ps_logger
from psrl.workers.ps.request_status_tracker import PSRL_RequestStatus

try:
    from psrl_state_grpc_proto import psrl_manager_pb2, psrl_manager_pb2_grpc
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Missing dependency `psrl-state-grpc-proto`. "
        "Please install it (for example: pip install -e psrl_state/python)."
    ) from exc

psrl_logger = get_ps_logger()


def _maybe_broadcast(values: list[Any], n: int, name: str) -> list[Any]:
    if len(values) == n:
        return values
    if len(values) == 1 and n > 1:
        return values * n
    raise ValueError(f"{name} length mismatch: got {len(values)}, expected 1 or {n}")


class PSManagerStateServicer(psrl_manager_pb2_grpc.PSManagerStateServicer):
    def __init__(self, ps_manager):
        self._ps = ps_manager

    @staticmethod
    def _handle_exception(context: grpc.ServicerContext, exc: Exception) -> None:
        psrl_logger.exception("PSManager gRPC call failed: %s", exc)
        context.set_code(grpc.StatusCode.INTERNAL)
        context.set_details(str(exc))

    def CanReserveRequest(self, request, context):
        try:
            request_ids = list(request.request_ids)
            model_versions = list(request.model_versions)
            if len(request_ids) == 0 or len(model_versions) == 0:
                return psrl_manager_pb2.CanReserveRequestResp(results=[], n_versions=len(model_versions))

            if len(request.is_validate) == 0:
                is_validate = False
            elif len(request.is_validate) == 1:
                is_validate = request.is_validate[0]
            else:
                is_validate = list(request.is_validate)

            results = self._ps.can_reserve_request(
                request_idx=request_ids,
                model_versions=model_versions,
                without_new_reserve_entry=request.without_new_reserve_entry,
                is_validate=is_validate,
            )

            flat_results: list[bool] = []
            if (
                len(request_ids) == 1
                and isinstance(results, list)
                and (len(results) == 0 or isinstance(results[0], bool))
            ):
                flat_results = [bool(v) for v in results]
            else:
                for row in results:
                    flat_results.extend(bool(v) for v in row)

            return psrl_manager_pb2.CanReserveRequestResp(
                results=flat_results,
                n_versions=len(model_versions),
            )
        except Exception as exc:  # noqa: BLE001
            self._handle_exception(context, exc)
            return psrl_manager_pb2.CanReserveRequestResp()

    def GetReserveIndicator(self, request, context):
        try:
            indicators = self._ps.get_reserve_indicator(
                request_id=request.request_id,
                model_versions=list(request.model_versions),
                is_validate=request.is_validate,
            )
            return psrl_manager_pb2.GetReserveIndicatorResp(indicators=indicators)
        except Exception as exc:  # noqa: BLE001
            self._handle_exception(context, exc)
            return psrl_manager_pb2.GetReserveIndicatorResp()

    def ReserveRolloutInstanceRequests(self, request, context):
        try:
            request_ids = list(request.request_ids)
            if len(request_ids) == 0:
                return psrl_manager_pb2.ReserveRolloutInstanceRequestsResp(success=True)

            instance_ids = [(iid.worker_id, iid.dp_rank) for iid in request.rollout_instance_ids]
            model_versions = list(request.model_versions)

            instance_ids = _maybe_broadcast(instance_ids, len(request_ids), "rollout_instance_ids")
            model_versions = _maybe_broadcast(model_versions, len(request_ids), "model_versions")

            buffer_ids, entry_ids = self._ps.reserve_rollout_instance_requests(
                rollout_instance_ids=instance_ids,
                request_ids=request_ids,
                model_versions=model_versions,
                guarantee_not_aborted=request.guarantee_not_aborted,
                is_validate=request.is_validate,
            )

            return psrl_manager_pb2.ReserveRolloutInstanceRequestsResp(
                success=True,
                buffer_ids=[-1 if v is None else int(v) for v in buffer_ids],
                entry_ids=[-1 if v is None else int(v) for v in entry_ids],
                error_message="",
            )
        except Exception as exc:  # noqa: BLE001
            self._handle_exception(context, exc)
            return psrl_manager_pb2.ReserveRolloutInstanceRequestsResp(
                success=False,
                error_message=str(exc),
            )

    def UpdateRequestInstanceId(self, request, context):
        try:
            new_instance_id = (request.new_instance_id.worker_id, request.new_instance_id.dp_rank)
            self._ps.update_request_instance_id(
                request_id=request.request_id,
                new_instance_id=new_instance_id,
                is_validate=request.is_validate,
            )
            return psrl_manager_pb2.UpdateRequestInstanceIdResp(success=True)
        except Exception as exc:  # noqa: BLE001
            self._handle_exception(context, exc)
            return psrl_manager_pb2.UpdateRequestInstanceIdResp(success=False)

    def UpdateRequestVersionTag(self, request, context):
        try:
            self._ps.update_request_version_tag(
                request_id=request.request_id,
                new_version_tag=request.new_version_tag,
                is_validate=request.is_validate,
            )
            return psrl_manager_pb2.UpdateRequestVersionTagResp(success=True)
        except Exception as exc:  # noqa: BLE001
            self._handle_exception(context, exc)
            return psrl_manager_pb2.UpdateRequestVersionTagResp(success=False)

    def CheckAbortedRequests(self, request, context):
        try:
            request_ids = list(request.request_ids)
            result = self._ps.check_aborted_requests(request_ids=request_ids, remove=request.remove)
            if isinstance(result, bool):
                result = [result]
            return psrl_manager_pb2.CheckAbortedRequestsResp(is_aborted=[bool(v) for v in result])
        except Exception as exc:  # noqa: BLE001
            self._handle_exception(context, exc)
            return psrl_manager_pb2.CheckAbortedRequestsResp()

    def CheckAbortedModelVersions(self, request, context):
        try:
            result = self._ps.check_aborted_model_versions(model_versions=list(request.model_versions))
            if isinstance(result, bool):
                result = [result]
            return psrl_manager_pb2.CheckAbortedModelVersionsResp(is_aborted=[bool(v) for v in result])
        except Exception as exc:  # noqa: BLE001
            self._handle_exception(context, exc)
            return psrl_manager_pb2.CheckAbortedModelVersionsResp()

    def GetRolloutInstanceModelVersion(self, request, context):
        try:
            rollout_instance_id = (
                request.rollout_instance_id.worker_id,
                request.rollout_instance_id.dp_rank,
            )
            model_version = self._ps.get_rollout_instance_model_version(rollout_instance_id=rollout_instance_id)
            return psrl_manager_pb2.GetRolloutInstanceModelVersionResp(model_version=model_version)
        except Exception as exc:  # noqa: BLE001
            self._handle_exception(context, exc)
            return psrl_manager_pb2.GetRolloutInstanceModelVersionResp(model_version=-1)

    def UpdateRequestStatus(self, request, context):
        try:
            request_ids = list(request.request_ids)
            if len(request_ids) == 0:
                return psrl_manager_pb2.UpdateRequestStatusResp(succeeded=[])

            status = PSRL_RequestStatus[request.status]
            model_versions = list(request.model_versions)
            rollout_instance_ids = [(iid.worker_id, iid.dp_rank) for iid in request.rollout_instance_ids]

            model_versions = _maybe_broadcast(model_versions or [-1], len(request_ids), "model_versions")
            rollout_instance_ids = _maybe_broadcast(
                rollout_instance_ids or [("-1", -1)],
                len(request_ids),
                "rollout_instance_ids",
            )

            updated = self._ps.update_request_status(
                request_id=request_ids,
                status=status,
                model_version=model_versions,
                rollout_instance_id=rollout_instance_ids,
                is_validate=request.is_validate,
            )
            if isinstance(updated, bool):
                updated = [updated]
            return psrl_manager_pb2.UpdateRequestStatusResp(succeeded=[bool(v) for v in updated])
        except Exception as exc:  # noqa: BLE001
            self._handle_exception(context, exc)
            return psrl_manager_pb2.UpdateRequestStatusResp(succeeded=[])


class PSManagerGrpcServer:
    def __init__(self, ps_manager):
        self._ps_manager = ps_manager
        self._server = None
        self._port = None

    def start(self, port: int = 0) -> int:
        if self._server is not None:
            return int(self._port)

        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=16),
            options=[
                # Permit 10-second client keepalives to avoid GOAWAY after two 30-second pings.
                ("grpc.http2.min_recv_ping_interval_without_data_ms", 10000),
                ("grpc.keepalive_permit_without_calls", True),
                # Rely on the receive interval instead of GOAWAY for ping rate limiting.
                ("grpc.http2.max_ping_strikes", 0),
            ],
        )
        psrl_manager_pb2_grpc.add_PSManagerStateServicer_to_server(
            PSManagerStateServicer(self._ps_manager),
            server,
        )
        bind_addr = f"[::]:{int(port)}"
        bound_port = server.add_insecure_port(bind_addr)
        if bound_port <= 0:
            raise RuntimeError(f"Failed to bind PSManager gRPC server at {bind_addr}")
        server.start()

        self._server = server
        self._port = int(bound_port)
        psrl_logger.info("PSManager gRPC server started on port %s", self._port)
        return self._port

    def stop(self, grace: float = 1.0) -> None:
        if self._server is None:
            return
        self._server.stop(grace)
        psrl_logger.info("PSManager gRPC server stopped")
        self._server = None
        self._port = None

    @property
    def port(self) -> int | None:
        return self._port
