# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Dexbotic Pi0 RL policy and loader live in the Dexbotic repo:
# ``dexbotic.rl.rlinf_bridge.dexbotic_pi0_policy``. Use ``ModelRegistry`` after
# ``dexbotic.rl.rlinf_registry.register_all()``.


def get_model(*args, **kwargs):
    raise RuntimeError(
        "Dexbotic Pi0 RL code was moved to dexbotic.rl.rlinf_bridge.dexbotic_pi0_policy. "
        "Import register_all from dexbotic.rl.rlinf_registry and call it before launching RLinf."
    )
