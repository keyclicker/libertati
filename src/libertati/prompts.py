"""Load user-editable model prompts from TOML."""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Prompts:
    """Prompt text used by the main agent and memory helper calls."""

    system: str
    roleplay: str
    web_search: str
    recall: str
    summary: str


def _required(data: dict[str, Any], section: str, key: str) -> str:
    """Return one required, non-empty prompt value."""
    try:
        value = data[section][key]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"missing prompt [{section}].{key}") from exc
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"prompt [{section}].{key} must be a non-empty string")
    return value


def load_prompts(path: Path) -> Prompts:
    """Read and validate every runtime prompt from ``path``."""
    with path.open("rb") as file:
        data = tomllib.load(file)
    return Prompts(
        system=_required(data, "agent", "system"),
        roleplay=_required(data, "agent", "roleplay"),
        web_search=_required(data, "agent", "web_search"),
        recall=_required(data, "memory", "recall"),
        summary=_required(data, "memory", "summary"),
    )
