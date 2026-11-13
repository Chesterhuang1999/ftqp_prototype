"""Diagnostics for static validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Diagnostic:
    message: str
    location: str = "<unknown>"

    def format(self) -> str:
        return f"{self.location}: {self.message}"


class ValidationError(Exception):
    def __init__(self, diagnostics: list[Diagnostic]) -> None:
        self.diagnostics = diagnostics
        super().__init__("\n".join(d.format() for d in diagnostics))
