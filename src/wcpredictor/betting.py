"""Betting evaluation: can the model beat the bookmaker?

The honest test of a forecasting model is not "is it better than guessing"
(that is what :mod:`wcpredictor.metrics` checks against a uniform baseline) but
"is it better than the *market*, by enough to overcome the bookmaker margin".

This module:
  * removes the bookmaker's margin ("de-vigs") to recover the market's implied
    fair probabilities;
  * scores the model against those market probabilities with log-loss;
  * walks a flat-stake and a fractional-Kelly bankroll through every match where
    the model thinks it has found a positive-expected-value (+EV) price.

Decimal odds throughout (e.g. 2.50 means stake 1 to win 1.50 profit).
Outcome index convention matches :mod:`wcpredictor.metrics`: 0=home, 1=draw, 2=away.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

Triple = Tuple[float, float, float]


def overround(odds: Triple) -> float:
    """The bookmaker's margin: sum of implied probabilities minus 1 (the 'vig')."""
    return sum(1.0 / o for o in odds) - 1.0


def devig(odds: Triple) -> Triple:
    """Fair (margin-free) probabilities from decimal odds, by proportional scaling."""
    inv = [1.0 / o for o in odds]
    s = sum(inv)
    return (inv[0] / s, inv[1] / s, inv[2] / s)


def devig_market(odds: Sequence[float]) -> List[float]:
    """Fair probabilities from ``N`` mutually-exclusive decimal odds.

    Generalises :func:`devig` (fixed to the 1X2 triple) to any number of outcomes,
    e.g. an outright-winner book. Proportional normalisation, so it handles both an
    overround (book sums to >1) and the underround you get from taking best-odds
    across bookmakers (sums to <1). Returns ``[]`` for an empty input.
    """
    inv = [1.0 / o if o > 0 else 0.0 for o in odds]
    s = sum(inv)
    if s <= 0:
        return [0.0 for _ in odds]
    return [x / s for x in inv]


def ev_per_unit(prob: float, dec_odds: float) -> float:
    """Expected profit per 1 unit staked on a back bet: p*odds - 1."""
    return prob * dec_odds - 1.0


def kelly_fraction(prob: float, dec_odds: float) -> float:
    """Full-Kelly stake fraction for a back bet; 0 if there is no edge."""
    b = dec_odds - 1.0
    if b <= 0:
        return 0.0
    f = (prob * b - (1.0 - prob)) / b
    return max(0.0, f)


@dataclass
class BetResult:
    n_matches: int
    n_bets: int
    model_log_loss: float        # model vs reality, on matches that have odds
    market_log_loss: float       # de-vigged market vs reality, same matches
    avg_overround: float         # mean bookmaker margin across those matches
    flat_staked: float           # 1 unit per +EV bet
    flat_profit: float
    flat_roi: float              # profit / staked
    kelly_start: float
    kelly_end: float             # bankroll after fractional-Kelly staking
    kelly_growth: float          # kelly_end / kelly_start - 1
    beats_market: bool           # model log-loss strictly below market log-loss


def evaluate(
    matches: Sequence[Tuple[Triple, Triple, int]],
    *,
    bet_probs: Sequence[Triple] | None = None,
    edge_threshold: float = 0.0,
    kelly_fraction_mult: float = 0.25,
    bankroll: float = 100.0,
    eps: float = 1e-12,
) -> BetResult:
    """Score model forecasts against market odds and backtest +EV staking.

    ``matches`` is a sequence of ``(model_probs, decimal_odds, outcome)`` tuples.
    Log-loss is always scored on those ``model_probs``. The *betting* decision can
    optionally use a different, parallel ``bet_probs`` sequence (e.g. shrunk toward
    the market) so the disciplined staking strategy is measured without distorting
    the accuracy comparison. A bet is placed on the single highest-EV selection per
    match when its edge exceeds ``edge_threshold``. Flat staking risks 1 unit per
    bet; the Kelly strategy risks ``kelly_fraction_mult`` of full Kelly against a
    running bankroll (default quarter-Kelly).
    """
    n = len(matches)
    if n == 0:
        return BetResult(0, 0, float("nan"), float("nan"), 0.0,
                         0.0, 0.0, 0.0, bankroll, bankroll, 0.0, False)

    model_ll = market_ll = vig_sum = 0.0
    n_bets = 0
    flat_staked = flat_profit = 0.0
    bank = bankroll

    for i, (probs, odds, outcome) in enumerate(matches):
        fair = devig(odds)
        model_ll += -math.log(max(probs[outcome], eps))
        market_ll += -math.log(max(fair[outcome], eps))
        vig_sum += overround(odds)

        # Pick the selection the (optionally shrunk) probabilities rate best-value.
        decide = bet_probs[i] if bet_probs is not None else probs
        evs = [ev_per_unit(decide[k], odds[k]) for k in range(3)]
        k = max(range(3), key=evs.__getitem__)
        if evs[k] <= edge_threshold:
            continue

        n_bets += 1
        won = outcome == k
        # Flat: 1 unit.
        flat_staked += 1.0
        flat_profit += (odds[k] - 1.0) if won else -1.0
        # Fractional Kelly against the live bankroll.
        stake = kelly_fraction(decide[k], odds[k]) * kelly_fraction_mult * bank
        bank += (odds[k] - 1.0) * stake if won else -stake

    model_ll /= n
    market_ll /= n
    return BetResult(
        n_matches=n,
        n_bets=n_bets,
        model_log_loss=model_ll,
        market_log_loss=market_ll,
        avg_overround=vig_sum / n,
        flat_staked=flat_staked,
        flat_profit=flat_profit,
        flat_roi=(flat_profit / flat_staked) if flat_staked else 0.0,
        kelly_start=bankroll,
        kelly_end=bank,
        kelly_growth=(bank / bankroll - 1.0) if bankroll else 0.0,
        beats_market=model_ll < market_ll,
    )
