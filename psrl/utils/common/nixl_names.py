"""
NIXL name string constants for PSRL.

This module is the single source of truth for all NIXL string identifiers
used to register agents and clients with the NIXL meta server. Both the NIXL
infrastructure layer (psrl.utils.nixl) and the worker naming layer
(psrl.utils.common.worker_naming) import from here.
"""

# NIXL coordination server name.
NIXL_META_SERVER_NAME = "NIXLMetaServer"

# Prefixes used to construct per-worker client names.
NIXL_GEN_CLIENT_PREFIX = "NIXLGenClient"
NIXL_TRAIN_CLIENT_PREFIX = "NIXLTrainClient"
NIXL_PS_CLIENT_PREFIX = "NIXLPSClient"
