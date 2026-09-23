"""Shared background-job provider contracts.

Keeping the handler type separate from discovery avoids circular imports when a
provider module is imported directly by tests or tooling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class JobHandler:
    """Name plus callable exported by one auto-discovered job provider."""

    name: str
    run: Callable[[str, str], object]
