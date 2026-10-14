# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hydra SearchPath plugin that falls back to veRL's config directory.

When PSRL doesn't have a local YAML override for a config group (e.g., actor/actor.yaml),
Hydra will automatically load it from veRL's trainer/config/ directory instead.
This eliminates the need to manually copy and maintain veRL YAML files in PSRL.
"""

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class PSRLSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # PSRL remains primary, so missing YAML groups fall back to this veRL path.
        search_path.append(
            provider="verl-base",
            path="pkg://verl.trainer.config",
        )
