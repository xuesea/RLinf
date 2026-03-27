# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dynamic model registry for external packages to register loaders without editing RLinf."""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any, Callable, Optional

from omegaconf import DictConfig, OmegaConf

_logger = logging.getLogger(__name__)


class ModelRegistry:
    """Maps model_type strings to (get_model_fn, metadata). Filled at runtime by callers (e.g. Dexbotic)."""

    _registry: dict[str, tuple[Callable[..., Any], dict]] = {}

    @classmethod
    def register(
        cls,
        model_type: str,
        get_model_fn: Callable[[DictConfig, Optional[Any]], Any],
        metadata: Optional[dict] = None,
        *,
        force: bool = False,
    ) -> None:
        if model_type in cls._registry and not force:
            raise ValueError(
                f"Model type {model_type!r} is already registered; "
                "pass force=True to replace the existing entry."
            )
        cls._registry[model_type] = (get_model_fn, metadata or {})

    @classmethod
    def get(cls, model_type: str) -> Optional[tuple[Callable[..., Any], dict]]:
        return cls._registry.get(model_type)

    @classmethod
    def list_registered(cls) -> list[str]:
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()


def ensure_model_registry_from_cfg(root_cfg: Optional[DictConfig]) -> None:
    """Load and run an external ``register_all()`` (e.g. Dexbotic) in Ray/subprocess workers.

    Main process registration does not propagate to Ray actors; set either
    ``runner.model_registry_init_module`` on the Hydra config or the environment variable
    ``RLINF_MODEL_REGISTRY_INIT_MODULE`` to the module path (e.g. ``dexbotic.rl.rlinf_registry``).
    """
    module_path: Optional[str] = None
    if root_cfg is not None:
        module_path = OmegaConf.select(
            root_cfg, "runner.model_registry_init_module", default=None
        )
    if not module_path:
        module_path = os.environ.get("RLINF_MODEL_REGISTRY_INIT_MODULE")
    if not module_path:
        return
    try:
        mod = importlib.import_module(module_path)
    except Exception:
        _logger.exception(
            "Failed to import model_registry_init_module %r", module_path
        )
        raise
    register_all = getattr(mod, "register_all", None)
    if register_all is not None:
        register_all()
