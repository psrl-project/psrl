"""
Optional tracing for elastic_rm router-backlog RPC chain (timeout drill-down).
"""

import logging
import os


def elastic_rm_backlog_diag_enabled() -> bool:
    """
    When true, emit WARNING logs for each hop in get_router_backlog_size → router.

    Enable with: ``PSRL_ELASTIC_RM_BACKLOG_DIAG=1`` (or ``true`` / ``yes``).
    """
    v = os.getenv("PSRL_ELASTIC_RM_BACKLOG_DIAG", "").strip().lower()
    return v in ("1", "true", "yes")


def log_elastic_rm_backlog_diag(logger: logging.Logger, msg: str, *args) -> None:
    if elastic_rm_backlog_diag_enabled():
        logger.warning("elastic_rm_diag " + msg, *args)
