"""Default simulated student profiles."""

from __future__ import annotations


def build_default_profiles() -> list[dict[str, str]]:
    """Return three simple profiles for the first dry-run demo."""
    return [
        {
            "profile": "Low",
            "profile_description": "A student with weak science knowledge who often gives vague or misconception-prone answers.",
        },
        {
            "profile": "Medium",
            "profile_description": "A student with basic science knowledge who can answer partly but may miss mechanisms or precision.",
        },
        {
            "profile": "High",
            "profile_description": "A student with strong science knowledge who usually gives concise, mechanism-based explanations.",
        },
    ]
