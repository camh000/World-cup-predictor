"""Shared pytest fixtures.

Every stochastic test takes an explicit seeded RNG so results are reproducible.
"""

import numpy as np
import pytest

from wcpredictor.config import Params, Paths
from wcpredictor.data_io import Team, read_seed_ratings, read_teams
from wcpredictor.ratings import Rating, RatingStore


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def params():
    return Params()


@pytest.fixture
def paths():
    return Paths()


@pytest.fixture
def teams(paths):
    return read_teams(paths.teams_csv)


@pytest.fixture
def ratings(paths, teams):
    seeds = read_seed_ratings(paths.seed_ratings_csv)
    return RatingStore.seed(teams, seeds)


@pytest.fixture
def tiny_group():
    """A single 4-team group for standings tests."""
    return [
        Team("AAA", "Alpha", "UEFA", "Z", False),
        Team("BBB", "Bravo", "UEFA", "Z", False),
        Team("CCC", "Charlie", "UEFA", "Z", False),
        Team("DDD", "Delta", "UEFA", "Z", False),
    ]


@pytest.fixture
def tiny_ratings():
    return RatingStore({
        "AAA": Rating(elo=1800),
        "BBB": Rating(elo=1700),
        "CCC": Rating(elo=1600),
        "DDD": Rating(elo=1500),
    })
