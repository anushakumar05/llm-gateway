from __future__ import annotations

from dataclasses import dataclass

from gateway.limits import LimitConfig


@dataclass
class Team:
    key: str
    limits: LimitConfig
    daily_budget_usd: float


# In real life this is a database. For the project, a dict is honest and clear.
TEAMS = {
    "team-normal": Team("team-normal", LimitConfig(rpm=60, tpm=100_000), daily_budget_usd=50.0),
    "team-cheap":  Team("team-cheap",  LimitConfig(rpm=10, tpm=10_000),  daily_budget_usd=1.0),
    "team-heavy":  Team("team-heavy",  LimitConfig(rpm=600, tpm=1_000_000), daily_budget_usd=500.0),
}

DEFAULT_TEAM = TEAMS["team-normal"]


def resolve_team(api_key: str | None) -> Team:
    return TEAMS.get(api_key or "", DEFAULT_TEAM)