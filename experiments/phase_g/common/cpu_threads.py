from __future__ import annotations

import os


THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def configure_cpu_threads_from_env(env_name: str = "OUTREACHLM_TARGET_CORES") -> dict[str, object]:
    raw_value = os.environ.get(env_name)
    if raw_value is None or raw_value.strip() == "":
        return {
            "target_cores": None,
            "source": "default",
            "env_vars": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
        }

    target = int(raw_value)
    if target <= 0:
        raise ValueError(f"{env_name} must be a positive integer when provided.")

    value = str(target)
    for name in THREAD_ENV_VARS:
        os.environ[name] = value

    return {
        "target_cores": target,
        "source": env_name,
        "env_vars": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
    }
