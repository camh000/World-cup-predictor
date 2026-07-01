"""Generate a gloriously early-2000s HTML dashboard from the live model.

Renders champion odds, group qualification, the friend-group sweepstake board,
and the model's recent report card into a single self-contained web/index.html
(no external assets) so it can be hosted anywhere as a static page.
"""

from __future__ import annotations

import csv
import importlib.util
from datetime import date, datetime
from pathlib import Path

from wcpredictor.config import Params, Paths
from wcpredictor.data_io import read_matches, read_seed_ratings, read_teams
from wcpredictor.betting import evaluate
from wcpredictor.history import read_predictions, summarize
from wcpredictor.ratings import RatingStore
from wcpredictor.scenarios import qualification
from wcpredictor.simulate import run_simulation
from wcpredictor.tournament import TeamStanding

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "index.html"   # repo root so Vercel/zero-config static serves it at "/"
N_SIMS = 20000

# Honest bet filter: don't act on noise. Shrink the model toward the market (the
# best available estimate) and only flag a "value" bet when a material edge
# survives. Applied to both the live value list and the realized scoreboard.
MARKET_SHRINK = 0.65    # 0 = pure model, 1 = pure market
MIN_EDGE = 0.10         # require >= 10% expected edge after shrinking
OUTRIGHT_MIN_EDGE = 0.20      # outright markets are noisier; demand a bigger edge
OUTRIGHT_MIN_MODEL_PROB = 0.03  # ignore longshot "value" (0.5% vs 0.1% at 1000/1 is noise)
# Dead-rubber rotation: in a round-3 game a team that has already QUALIFIED or is
# OUT often rests players. Regress its effective Elo this fraction toward the
# field mean to model the extra uncertainty. 0 disables it.
ROTATION_DAMP = 0.30


def _load_friends():
    spec = importlib.util.spec_from_file_location("sweepstakes", Path(__file__).with_name("sweepstakes.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FRIENDS, mod.EXTRA_ALIASES


def _hits() -> int:
    # A "visitor counter" that creeps up over time, for that authentic feel.
    return 13337 + (date.today() - date(2026, 6, 11)).days * 391


def _favicon() -> str:
    """A 16x16 pixel-art football as an SVG data URI (crisp, no external file)."""
    import urllib.parse

    n, c, r = 16, 7.5, 7.0
    px = [["."] * n for _ in range(n)]
    for y in range(n):
        for x in range(n):
            d = ((x - c) ** 2 + (y - c) ** 2) ** 0.5
            if d <= r:
                px[y][x] = "w"
            if r - 1.3 <= d <= r:
                px[y][x] = "k"               # black outline ring
    for y in range(n):                        # central black pentagon (diamond blob)
        for x in range(n):
            if abs(x - c) + abs(y - c) <= 2.2 and px[y][x] != ".":
                px[y][x] = "k"
    for bx, by in [(4, 4), (11, 4), (4, 11), (11, 11), (7, 13)]:   # surrounding patches
        for y in range(by - 1, by + 1):
            for x in range(bx - 1, bx + 1):
                if 0 <= x < n and 0 <= y < n and px[y][x] == "w":
                    px[y][x] = "k"
    rects = "".join(
        f'<rect x="{x}" y="{y}" width="1" height="1" fill="{"#fff" if px[y][x]=="w" else "#000"}"/>'
        for y in range(n) for x in range(n) if px[y][x] != "."
    )
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {n} {n}" '
           f'shape-rendering="crispEdges">{rects}</svg>')
    return "data:image/svg+xml," + urllib.parse.quote(svg)


def _cursor() -> str:
    """A little round football as an SVG data-URI mouse cursor (hotspot at centre)."""
    import urllib.parse

    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28">'
        '<circle cx="14" cy="14" r="12" fill="#ffffff" stroke="#000000" stroke-width="2"/>'
        '<polygon points="14,8 18,11 16,16 12,16 10,11" fill="#000000"/>'
        '<line x1="14" y1="8" x2="14" y2="2.5" stroke="#000" stroke-width="1.6"/>'
        '<line x1="18" y1="11" x2="23.5" y2="9" stroke="#000" stroke-width="1.6"/>'
        '<line x1="16" y1="16" x2="19.5" y2="21.5" stroke="#000" stroke-width="1.6"/>'
        '<line x1="12" y1="16" x2="8.5" y2="21.5" stroke="#000" stroke-width="1.6"/>'
        '<line x1="10" y1="11" x2="4.5" y2="9" stroke="#000" stroke-width="1.6"/>'
        '</svg>'
    )
    return "data:image/svg+xml," + urllib.parse.quote(svg)


def _load_ratings(paths, teams):
    """Load learned ratings, or fall back to the committed seed ratings.

    Never silently fall back to a flat ~1500 table (the old ``seed(teams, {})``
    produced a plausible-looking but garbage dashboard with every team equal): if
    neither learned state nor seed ratings exist, fail loudly so a missing or
    partial ``state/`` is obvious rather than rendered as real predictions.
    """
    if paths.ratings_json.exists():
        return RatingStore.load(paths.ratings_json)
    seeds = read_seed_ratings(paths.seed_ratings_csv)
    if not seeds:
        raise SystemExit(
            f"error: no learned ratings at {paths.ratings_json} and no seed "
            f"ratings at {paths.seed_ratings_csv}; run 'wcpredict reset' then "
            f"'import-results' + 'replay' before generating the dashboard."
        )
    print(f"note: {paths.ratings_json} missing; using seed ratings "
          f"({len(seeds)} teams) from {paths.seed_ratings_csv}.")
    return RatingStore.seed(teams, seeds)


def main() -> None:
    paths = Paths()
    teams = read_teams(paths.teams_csv)
    params = Params.load(paths.params_json)
    ratings = _load_ratings(paths, teams)
    name = {t.team_id: t.name for t in teams}

    df = run_simulation(teams, params, ratings, n_sims=N_SIMS, seed=42)
    pchamp = {r.team_id: r.p_champion for r in df.itertuples()}

    group_matches = [m for m in read_matches(paths.results_csv) if m.stage == "group"]
    base, adv, win = qualification(teams, group_matches, params, ratings, n_sims=8000, seed=42)

    preds = read_predictions(paths.predictions_csv)
    summary = summarize(preds)

    friends, aliases = _load_friends()
    from wcpredictor.data_io import _norm_name, build_name_index
    idx = build_name_index(teams)
    idx.update({_norm_name(k): v for k, v in aliases.items()})
    friend_rows = []
    for person, tt in friends.items():
        total = sum(pchamp.get(idx.get(_norm_name(x), ""), 0.0) for x in tt)
        best = max(tt, key=lambda x: pchamp.get(idx.get(_norm_name(x), ""), 0.0))
        friend_rows.append((person, total, best))
    friend_rows.sort(key=lambda r: -r[1])

    # Calibrate the model's probabilities (temperature, fit on settled games).
    from wcpredictor.calibration import fit_sharpness, sharpen
    from wcpredictor.metrics import log_loss as _ll
    om = {"home": 0, "draw": 1, "away": 2}
    cal_p = [(float(r["p_home"]), float(r["p_draw"]), float(r["p_away"]))
             for r in preds if r["actual_outcome"] in om]
    cal_o = [om[r["actual_outcome"]] for r in preds if r["actual_outcome"] in om]
    gamma = fit_sharpness(cal_p, cal_o) if cal_p else 1.0
    ll_raw = _ll(cal_p, cal_o) if cal_p else float("nan")
    ll_cal = _ll([sharpen(p, gamma) for p in cal_p], cal_o) if cal_p else float("nan")
    calib = (gamma, ll_raw, ll_cal)

    # Mathematical (not just probable) group fate, from results played so far.
    played_pairs = {frozenset((m.home_team_id, m.away_team_id)) for m in group_matches}
    clinch = _clinch_status(base, adv, played_pairs)

    upcoming = _upcoming(paths, teams, params, ratings, name, gamma, clinch, n=72)
    bet = _betting(paths, preds, gamma)
    clv_data = _clv_scoreboard(paths, preds, gamma)
    outright = _load_outright(paths)

    bracket_html = _knockout_bracket(paths, base, params, ratings, name)

    html = _render(df, name, base, adv, win, preds, summary, friend_rows,
                   upcoming, bet, calib, outright, clinch, clv_data, bracket_html)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT}")


def _load_odds(paths):
    """Bookmaker decimal odds keyed by (home_id, away_id) from data/odds.csv."""
    import csv

    odds_path = paths.data_dir / "odds.csv"
    out = {}
    if not odds_path.exists():
        return out
    with odds_path.open("r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            out[(r["home_team_id"], r["away_team_id"])] = (
                float(r["odds_home"]), float(r["odds_draw"]), float(r["odds_away"]))
    return out


def _load_outright(paths):
    """Outright 'winner' decimal odds keyed by team_id from data/outright_odds.csv."""
    import csv

    p = paths.data_dir / "outright_odds.csv"
    out = {}
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            out[r["team_id"]] = float(r["odds_decimal"])
    return out


def _upcoming(paths, teams, params, ratings, name, gamma=1.0, clinch=None, n=40):
    """Rich forecasts for the next ``n`` unplayed group-stage fixtures.

    Each row is a dict with the model's (calibrated) probabilities and most-likely
    score, plus — where we hold bookmaker odds — the de-vigged market probabilities
    and the model's best +EV ("value") selection against that price. For round-3
    games, a team that has already QUALIFIED or is OUT (per ``clinch``) has its
    effective Elo regressed toward the field mean to model likely rotation.
    """
    import csv
    import numpy as np
    from wcpredictor.betting import devig, ev_per_unit
    from wcpredictor.calibration import blend, sharpen
    from wcpredictor.data_io import HOST_TEAM_IDS, _norm_name, build_name_index
    from wcpredictor.history import forecast
    from wcpredictor.poisson import dixon_coles_matrix

    clinch = clinch or {}
    mean_elo = sum(ratings[t.team_id].elo for t in teams) / len(teams)

    def _rotation_adj(team_id, rnd):
        # Dead rubber: a clinched/eliminated team in round 3 likely rests players.
        if rnd == "3" and clinch.get(team_id) in ("QUALIFIED", "OUT"):
            return (mean_elo - ratings[team_id].elo) * ROTATION_DAMP
        return 0.0

    idx = build_name_index(teams)
    odds_map = _load_odds(paths)
    # Fixtures already played (recorded in results.csv) must not show as "upcoming",
    # even if the schedule's own Result column hasn't been filled in.
    played = set()
    res_path = paths.data_dir / "results.csv"
    if res_path.exists():
        with res_path.open("r", encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                played.add(frozenset((r["home_team_id"], r["away_team_id"])))
    sched = paths.data_dir / "fifa_worldcup_2026_schedule.csv"
    rows = []
    with sched.open("r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if (r.get("Result") or "").strip():
                continue
            rnd = r.get("Round Number", "").strip()
            if rnd not in {"1", "2", "3"}:
                continue
            h = idx.get(_norm_name(r.get("Home Team", "")))
            a = idx.get(_norm_name(r.get("Away Team", "")))
            if not h or not a:
                continue
            if frozenset((h, a)) in played:
                continue
            try:
                dt = datetime.strptime(r["Date"].strip(), "%d/%m/%Y %H:%M")
            except ValueError:
                continue
            rows.append((dt, h, a, rnd))
    rows.sort(key=lambda x: x[0])

    out = []
    for dt, h, a, rnd in rows[:n]:
        neutral = h not in HOST_TEAM_IDS
        adj_h, adj_a = _rotation_adj(h, rnd), _rotation_adj(a, rnd)
        rotate = [t for t, adj in ((h, adj_h), (a, adj_a)) if adj != 0.0]
        probs_raw, (lh, la) = forecast(ratings, params, h, a, neutral,
                                       adj_home=adj_h, adj_away=adj_a)
        probs = sharpen(probs_raw, gamma)
        m = dixon_coles_matrix(lh, la, params.dc_rho, params.max_goals)
        i, j = np.unravel_index(int(np.argmax(m)), m.shape)
        row = {"dt": dt, "h": h, "a": a, "p": probs, "p_raw": probs_raw,
               "score": (int(i), int(j)), "rotate": rotate,
               "odds": None, "mkt": None, "sel": None, "edge": None}
        odds = odds_map.get((h, a))
        if odds:
            mkt = devig(odds)
            # Bet on the model shrunk toward the market, not the raw model: this
            # filters the small, noisy "edges" the model invents on most games.
            decide = blend(probs, mkt, MARKET_SHRINK)
            evs = [ev_per_unit(decide[k], odds[k]) for k in range(3)]
            sel = max(range(3), key=evs.__getitem__)
            row.update(odds=odds, mkt=mkt, sel=sel, edge=evs[sel])
        out.append(row)
    return out


def _betting(paths, preds, gamma=1.0):
    """Join settled predictions to bookmaker odds (data/odds.csv) and backtest.

    Accuracy (log-loss) is scored on the pure model; the staking decision uses the
    same disciplined filter as the live list — shrink toward the market, require a
    material edge — so the scoreboard measures the strategy we would actually run.
    """
    import csv
    from wcpredictor.betting import devig
    from wcpredictor.calibration import blend, sharpen

    odds_path = paths.data_dir / "odds.csv"
    if not odds_path.exists():
        return None
    idx = {}
    with odds_path.open("r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            idx[(r["date"], r["home_team_id"], r["away_team_id"])] = (
                float(r["odds_home"]), float(r["odds_draw"]), float(r["odds_away"]))
    om = {"home": 0, "draw": 1, "away": 2}
    matches, bet_probs = [], []
    for r in preds:
        k = (r["date"], r["home_team_id"], r["away_team_id"])
        if k in idx and r["actual_outcome"] in om:
            probs = sharpen((float(r["p_home"]), float(r["p_draw"]), float(r["p_away"])), gamma)
            matches.append((probs, idx[k], om[r["actual_outcome"]]))
            bet_probs.append(blend(probs, devig(idx[k]), MARKET_SHRINK))
    if not matches:
        return None
    return evaluate(matches, bet_probs=bet_probs, edge_threshold=MIN_EDGE)


def _clv_scoreboard(paths, preds, gamma=1.0):
    """Closing-line-value scoreboard from data/odds_history.csv.

    For each fixture with >=2 timestamped odds snapshots, the earliest snapshot is
    the price the model would have 'taken' (when it flagged the bet) and the latest
    is the closing line. For settled fixtures we replay the same honest filter
    (shrink to the *opening* market, require MIN_EDGE) and measure whether the bet
    beat the close (CLV = taken_odds * closing_fair_prob - 1). Beating the close is
    the real proof of edge -- independent of whether the bet won. Returns counts
    (so the panel can show it accumulating) plus the per-bet CLV rows.
    """
    import csv
    from collections import defaultdict
    from wcpredictor.betting import clv, devig, ev_per_unit
    from wcpredictor.calibration import blend, sharpen

    snaps = defaultdict(list)
    timestamps = set()
    hist = paths.data_dir / "odds_history.csv"
    if hist.exists():
        with hist.open("r", encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                timestamps.add(r["fetched_at"])
                snaps[(r["date"], r["home_team_id"], r["away_team_id"])].append(
                    (r["fetched_at"], (float(r["odds_home"]), float(r["odds_draw"]), float(r["odds_away"]))))

    om = {"home": 0, "draw": 1, "away": 2}
    pred_idx = {}
    for r in preds:
        if r["actual_outcome"] in om:
            pred_idx[(r["date"], r["home_team_id"], r["away_team_id"])] = (
                (float(r["p_home"]), float(r["p_draw"]), float(r["p_away"])), om[r["actual_outcome"]])

    n_pairs = sum(1 for v in snaps.values() if len({t for t, _ in v}) >= 2)
    bets = []
    for k, v in snaps.items():
        ts = sorted({t for t, _ in v})
        if len(ts) < 2 or k not in pred_idx:
            continue
        open_o = next(o for t, o in v if t == ts[0])
        close_o = next(o for t, o in v if t == ts[-1])
        probs, actual = pred_idx[k]
        decide = blend(sharpen(probs, gamma), devig(open_o), MARKET_SHRINK)
        evs = [ev_per_unit(decide[i], open_o[i]) for i in range(3)]
        sel = max(range(3), key=evs.__getitem__)
        if evs[sel] <= MIN_EDGE:
            continue
        cval = clv(open_o[sel], devig(close_o)[sel])
        won = actual == sel
        bets.append({"fix": k, "sel": sel, "open": open_o[sel], "close": close_o[sel],
                     "clv": cval, "won": won, "pl": (open_o[sel] - 1.0) if won else -1.0})
    return {"n_snapshots": len(timestamps), "first": min(timestamps)[:10] if timestamps else None,
            "last": max(timestamps)[:10] if timestamps else None,
            "n_tracked": len(snaps), "n_pairs": n_pairs, "bets": bets}


# --------------------------------------------------------------------------- #
# Rendering (intentionally retro)
# --------------------------------------------------------------------------- #
def _clinch_status(base, adv, played_pairs):
    """Per-team group fate that is *mathematically* true, not just probable.

    For each group we enumerate every remaining group result (each pairing is
    win/draw/loss; <=81 combos) and judge on points only, worst-case for ties:

      * QUALIFIED -> in *every* scenario at most one other team can reach the
        team's points, so it is guaranteed a top-2 finish (which always advances)
        no matter the goal-difference tiebreaks;
      * OUT -> in *every* scenario at least two other teams finish strictly above
        it (so top-2 is impossible) AND the simulation gives it ~no best-third
        hope either;
      * otherwise None -> show the simulated probability, never a false promise.

    Once the group stage is COMPLETE the tables are final, so every team is
    resolved to QUALIFIED/OUT outright (the top two per group plus the eight best
    third-placed teams, by the full tiebreakers) -- no team should still read as a
    probability once its group is done.
    """
    from itertools import combinations, product

    if group_stage_complete(base):
        from wcpredictor.tournament import N_BEST_THIRDS, best_third_placed
        status, thirds = {}, []
        for g, stands in base.items():
            table = sorted(stands.values(), key=lambda s: s.sort_key())
            status[table[0].team_id] = "QUALIFIED"
            status[table[1].team_id] = "QUALIFIED"
            for s in table[3:]:
                status[s.team_id] = "OUT"
            if len(table) >= 3:
                thirds.append(table[2])
        best_ids = {s.team_id for s in best_third_placed(thirds, N_BEST_THIRDS)}
        for s in thirds:
            status[s.team_id] = "QUALIFIED" if s.team_id in best_ids else "OUT"
        return status

    status = {}
    for g, stands in base.items():
        tids = list(stands)
        pts = {t: stands[t].points for t in tids}
        remaining = [tuple(p) for p in combinations(tids, 2)
                     if frozenset(p) not in played_pairs]
        clinched = {t: True for t in tids}
        eliminated = {t: True for t in tids}
        for sc in product((0, 1, 2), repeat=len(remaining)):
            fp = dict(pts)
            for (a, b), o in zip(remaining, sc):
                if o == 0:
                    fp[a] += 3
                elif o == 1:
                    fp[a] += 1
                    fp[b] += 1
                else:
                    fp[b] += 3
            for t in tids:
                if sum(1 for u in tids if u != t and fp[u] >= fp[t]) > 1:
                    clinched[t] = False
                if sum(1 for u in tids if u != t and fp[u] > fp[t]) < 2:
                    eliminated[t] = False
        for t in tids:
            if clinched[t]:
                status[t] = "QUALIFIED"
            elif eliminated[t] and adv.get(t, 0.0) <= 0.0005:
                status[t] = "OUT"
            else:
                status[t] = None
    return status


def group_stage_complete(base) -> bool:
    """True once every team has played all three of its group games."""
    return bool(base) and all(
        s.played >= 3 for tbl in base.values() for s in tbl.values())


def _knockout_bracket(paths, base, params, ratings, name) -> str:
    """Knockout tracker built from the REAL data, not a reconstructed bracket.

    Shows played knockout games (data/results.csv, stage != 'group') with their
    scores and the side that went through, then the actual upcoming knockout ties
    (the priced fixtures in data/odds.csv, once the group stage is over) with the
    model's knockout win probability. Returns "" until the knockouts start.

    We deliberately do NOT rebuild the bracket from the model's own third-place
    allocation: it mis-slots the eight best thirds (it paired Germany with Bosnia
    when the real draw sent Germany to Paraguay), so the results file is the single
    source of truth and this updates itself as games are played.
    """
    from wcpredictor.history import forecast

    all_matches = read_matches(paths.results_csv)
    ko = sorted((m for m in all_matches if m.stage != "group"), key=lambda m: m.date)
    if not ko and not group_stage_complete(base):
        return ""   # still in the group stage; the UPCOMING panel covers those
    played = {frozenset((m.home_team_id, m.away_team_id)) for m in all_matches}
    upcoming = [(h, a) for (h, a) in _load_odds(paths) if frozenset((h, a)) not in played]
    if not ko and not upcoming:
        return ""

    def _grid(boxes, cols=4):
        cells = "".join(f'<td valign="top">{b}</td>' + ("</tr><tr>" if (i + 1) % cols == 0 else "")
                        for i, b in enumerate(boxes))
        return f'<table cellpadding="6"><tr>{cells}</tr></table>'

    def result_box(m):
        hg, ag = m.home_goals, m.away_goals
        winner = None if hg == ag else (m.home_team_id if hg > ag else m.away_team_id)
        def cell(tid, goals):
            through = winner == tid
            nm = f"<b>{name.get(tid, tid)}</b>" if through else name.get(tid, tid)
            arrow = "&#9656;" if through else "&nbsp;"
            return (f'<tr bgcolor="{"#CCFFCC" if through else "#FFFFFF"}"><td>{arrow}&nbsp;{nm}</td>'
                    f'<td align="center"><b>{goals}</b></td></tr>')
        note = " &middot; pens" if winner is None else ""
        return ('<table border="1" cellpadding="2" cellspacing="0" width="210" bgcolor="#FFFFFF">'
                f'<tr bgcolor="#006400"><td colspan="2"><font color="#FFFF00" size="1">'
                f'&nbsp;{m.stage.title()} &middot; {m.date}{note}</font></td></tr>'
                f'{cell(m.home_team_id, hg)}{cell(m.away_team_id, ag)}</table>')

    def predict_box(h, a):
        probs, _ = forecast(ratings, params, h, a, True)
        ph, pa = probs[0] + 0.5 * probs[1], probs[2] + 0.5 * probs[1]
        home = ph >= pa
        def cell(tid, p, win):
            nm = f"<b>{name.get(tid, tid)}</b>" if win else name.get(tid, tid)
            arrow = "&#9656;" if win else "&nbsp;"
            return (f'<tr bgcolor="{"#CCFFCC" if win else "#FFFFFF"}"><td>{arrow}&nbsp;{nm}</td>'
                    f'<td align="center"><font size="1">{p*100:.0f}%</font></td></tr>')
        return ('<table border="1" cellpadding="2" cellspacing="0" width="210" bgcolor="#FFFFFF">'
                '<tr bgcolor="#000080"><td colspan="2"><font color="#FFFF00" size="1">'
                '&nbsp;Next up</font></td></tr>'
                f'{cell(h, ph, home)}{cell(a, pa, not home)}</table>')

    out = '<h2><font color="#FFFFFF">&#127942; KNOCKOUT STAGE</font></h2>'
    if ko:
        out += ('<h3><font color="#FFFFFF">Results so far</font></h3>'
                '<p><font size="1" face="Courier New">Real scores from the results file; the side '
                'that went through is highlighted (a level tie was decided on penalties).</font></p>'
                + _grid([result_box(m) for m in ko]))
    if upcoming:
        out += ('<h3><font color="#FFFFFF">Coming up</font></h3>'
                '<p><font size="1" face="Courier New">The actual next ties (as priced in the odds '
                'feed). Each box shows the model&rsquo;s knockout win probability (draws split 50/50 '
                'for extra time / penalties); likely winner highlighted.</font></p>'
                + _grid([predict_box(h, a) for h, a in upcoming]))
    return out


def _fmt_date(iso: str | None) -> str | None:
    """ISO 'YYYY-MM-DD' -> human '20 Jun 2026' (None passes through)."""
    if not iso:
        return None
    try:
        d = datetime.strptime(iso[:10], "%Y-%m-%d")
    except ValueError:
        return iso
    return f"{d.day} {d:%b %Y}"


def _render(df, name, base, adv, win, preds, summary, friend_rows, upcoming, bet, calib, outright, clinch, clv_data, bracket_html="") -> str:
    updated = datetime.utcnow().strftime("%A %d %B %Y, %H:%M UTC")
    fav = _favicon()
    cur = _cursor()

    from wcpredictor.betting import devig_market
    _ids = list(outright)
    mkt_champ = dict(zip(_ids, devig_market([outright[t] for t in _ids])))

    # Odds source/date, derived from the data so captions never hard-code a stale
    # date or provider once the live snapshots replace the committed sample.
    odds_asof = _fmt_date(clv_data.get("last") or clv_data.get("first"))
    odds_src = (f"real bookmaker odds, as of {odds_asof}" if odds_asof
                else "real bookmaker odds")
    outright_book_pct = (sum(1.0 / o for o in outright.values()) * 100
                         if outright else None)

    # Softer market: outright-winner value. Shrink the model's champion prob toward
    # the (de-vigged) market, then back the raw outright price when a big edge
    # survives. Outrights are less efficient than 1X2, so we demand a larger edge.
    champ_model = {r.team_id: float(r.p_champion) for r in df.itertuples()}
    outright_value = []
    for t, o in outright.items():
        mp, mk = champ_model.get(t, 0.0), mkt_champ.get(t, 0.0)
        if mp < OUTRIGHT_MIN_MODEL_PROB:   # skip longshot noise
            continue
        edge = ((1.0 - MARKET_SHRINK) * mp + MARKET_SHRINK * mk) * o - 1.0
        if edge > OUTRIGHT_MIN_EDGE:
            outright_value.append((t, o, mp, mk, edge))
    outright_value.sort(key=lambda x: -x[4])
    if outright_value:
        _orows = "".join(
            f'<tr bgcolor="{"#FFFFCC" if i % 2 else "#FFFFFF"}">'
            f'<td>&nbsp;{name.get(t, t)}</td><td align="center">{o:.0f}</td>'
            f'<td align="center">{mp*100:.1f}%</td><td align="center">{mk*100:.1f}%</td>'
            f'<td align="center"><b><font color="#008000">+{edge*100:.0f}%</font></b></td></tr>'
            for i, (t, o, mp, mk, edge) in enumerate(outright_value[:6]))
        outright_html = (
            '<table border="2" cellpadding="3" cellspacing="0" bgcolor="#FFFFFF" width="70%">'
            '<tr bgcolor="#006400"><th align="left"><font color="#FFFF00">&nbsp;Team to win it</font></th>'
            '<th><font color="#FFFF00">Odds</font></th><th><font color="#FFFF00">Model</font></th>'
            '<th><font color="#FFFF00">Market</font></th><th><font color="#FFFF00">Edge</font></th></tr>'
            f'{_orows}</table>'
            f'<p><font size="1" face="Courier New">Outright winner market &#8212; less efficient than match '
            f'odds, so the model&rsquo;s remaining top-heaviness can masquerade as value here too. Same health '
            f'warning applies.</font></p>')
    else:
        outright_html = (f'<p>No outright-winner bet clears the &ge;{int(OUTRIGHT_MIN_EDGE*100)}% edge bar '
                         f'after shrinking toward the market.</p>')

    # Data-driven champion caption: compare the model's favourite to the market so
    # the text can never go stale against the table above it.
    fav_row = df.iloc[0]
    fav_id, fav_model = fav_row["team_id"], float(fav_row["p_champion"])
    fav_name = name.get(fav_id, fav_id)
    fav_mkt = mkt_champ.get(fav_id)
    if fav_mkt is not None and fav_model > fav_mkt + 0.05:
        champ_lean = (f'The model still leans a little top-heavy &#8212; it makes {fav_name} favourite at '
                      f'<b>{fav_model*100:.0f}%</b> vs the market&rsquo;s <b>{fav_mkt*100:.0f}%</b>, but that is '
                      f'now a believable disagreement (it was a silly ~31% vs ~10% before the calibration fix), '
                      f'not a glitch. It still ranks by raw Elo, where {fav_name} is genuinely top.')
    elif fav_mkt is not None:
        champ_lean = (f'The model now broadly tracks the winner market &#8212; {fav_name} favourite at '
                      f'<b>{fav_model*100:.0f}%</b> vs the market&rsquo;s <b>{fav_mkt*100:.0f}%</b>.')
    else:
        champ_lean = (f'The model makes {fav_name} favourite at <b>{fav_model*100:.0f}%</b>.')
    asof = f' (as of {odds_asof})' if odds_asof else ''
    book_sum = (f' Outright &quot;best odds&quot; sum to ~{outright_book_pct:.0f}%, so the de-vig '
                f'is approximate.' if outright_book_pct else '')
    champ_caption = (f'Model vs bookies&rsquo; winner market{asof}. {champ_lean}{book_sum}')
    champ_rows = ""
    for i, r in enumerate(df.head(12).itertuples()):
        m = mkt_champ.get(r.team_id)
        mcell = f'{m*100:.1f}%' if m is not None else '&ndash;'
        champ_rows += (f'<tr bgcolor="{"#FFFFCC" if i % 2 else "#FFFFFF"}">'
                       f'<td align="center"><b>{i+1}</b></td><td>&nbsp;{r.team}</td>'
                       f'<td align="center">{r.elo:.0f}</td>'
                       f'<td align="center"><b>{r.p_champion*100:.1f}%</b></td>'
                       f'<td align="center">{mcell}</td></tr>')

    group_blocks = []
    for g in sorted(base):
        table = sorted(base[g].values(), key=lambda s: s.sort_key())
        rows = ""
        for s in table:
            a = adv[s.team_id]
            st = clinch.get(s.team_id)
            if st == "QUALIFIED":
                colour, tag = "#CCFFCC", "QUALIFIED!"
            elif st == "OUT":
                colour, tag = "#FFCCCC", "OUT"
            else:
                colour = "#FFFFFF"
                tag = ">99%" if a >= 0.995 else "&lt;1%" if a <= 0.005 else f"{a*100:.0f}%"
            rows += (f'<tr bgcolor="{colour}"><td>&nbsp;{name[s.team_id]}</td>'
                     f'<td align="center">{s.points}</td><td align="center">{s.gd:+d}</td>'
                     f'<td align="center"><b>{tag}</b></td></tr>')
        group_blocks.append(
            f'<table border="2" cellpadding="3" cellspacing="0" width="240" bgcolor="#FFFFFF">'
            f'<tr bgcolor="#000080"><th colspan="4"><font color="#FFFF00">GROUP {g}</font></th></tr>'
            f'<tr bgcolor="#C0C0C0"><th align="left">&nbsp;Team</th><th>Pts</th><th>GD</th><th>Adv</th></tr>'
            f'{rows}</table>')
    groups_html = "".join(f'<td valign="top">{b}&nbsp;</td>'
                          + ("</tr><tr>" if (i + 1) % 4 == 0 else "")
                          for i, b in enumerate(group_blocks))

    sweep_rows = ""
    medals = {0: "🥇", 1: "🥈", 2: "🥉"}
    for i, (person, total, best) in enumerate(friend_rows):
        sweep_rows += (f'<tr bgcolor="{"#FFFFCC" if i % 2 else "#FFFFFF"}">'
                       f'<td align="center">{medals.get(i, i+1)}</td><td>&nbsp;{person}</td>'
                       f'<td align="center"><b>{total*100:.1f}%</b></td><td>&nbsp;{best}</td></tr>')

    pred_rows = ""
    for r in preds[-10:][::-1]:
        hit = r["predicted_outcome"] == r["actual_outcome"]
        pred_rows += (f'<tr bgcolor="{"#CCFFCC" if hit else "#FFCCCC"}">'
                      f'<td>&nbsp;{r["home_team_id"]} {r["home_goals"]}-{r["away_goals"]} {r["away_team_id"]}</td>'
                      f'<td align="center">{r["predicted_outcome"]}</td>'
                      f'<td align="center">{"HIT" if hit else "miss"}</td></tr>')
    def _sel_label(row, k):
        return name[row["h"]] if k == 0 else ("Draw" if k == 1 else name[row["a"]])

    up_rows = ""
    any_rotation = False
    for r in upcoming:
        ph, pd, pa = r["p"]
        si, sj = r["score"]
        pick = name[r["h"]] if ph >= max(pd, pa) else (name[r["a"]] if pa >= pd else "Draw")
        rot = set(r.get("rotate") or [])
        any_rotation = any_rotation or bool(rot)
        hn = name[r["h"]] + ("&dagger;" if r["h"] in rot else "")
        an = name[r["a"]] + ("&dagger;" if r["a"] in rot else "")
        if r["odds"] is None:
            bet_cell = '<font color="#888888">no odds</font>'
        elif r["edge"] > MIN_EDGE:
            bet_cell = f'<font color="#B8860B">{_sel_label(r, r["sel"])} +{r["edge"]*100:.0f}%</font>'
        else:
            bet_cell = '<font color="#888888">agrees</font>'
        up_rows += (f'<tr bgcolor="{"#FFFFCC" if len(up_rows) % 2 else "#FFFFFF"}">'
                    f'<td>&nbsp;{r["dt"]:%a %d %b %H:%M}Z</td>'
                    f'<td>&nbsp;{hn} v {an}</td>'
                    f'<td align="center">{ph*100:.0f} / {pd*100:.0f} / {pa*100:.0f}</td>'
                    f'<td align="center"><b>{pick}</b></td>'
                    f'<td align="center">{si}-{sj}</td>'
                    f'<td align="center">{bet_cell}</td></tr>')

    rot_note = (" &dagger; = likely to rotate (already qualified or eliminated), so its rating is "
                "regressed toward the field." if any_rotation else "")

    # The "remaining group-stage matches" table empties out once the group stage
    # finishes; drop the whole section then (the knockout bracket takes over above).
    if up_rows:
        upcoming_section = (
            '<h2><font color="#FFFFFF">&#128302; REMAINING GROUP-STAGE MATCHES &#8212; PREDICTED</font></h2>'
            '<table border="2" cellpadding="3" cellspacing="0" bgcolor="#FFFFFF" width="75%">'
            '<tr bgcolor="#000080"><th align="left"><font color="#FFFF00">&nbsp;Kickoff (UTC)</font></th>'
            '<th align="left"><font color="#FFFF00">&nbsp;Match</font></th>'
            '<th><font color="#FFFF00">Win / Draw / Win %</font></th>'
            '<th><font color="#FFFF00">Tip</font></th><th><font color="#FFFF00">xScore</font></th>'
            '<th><font color="#FFFF00">Model vs bookies*</font></th></tr>'
            f'{up_rows}</table>'
            '<p><font size="1" face="Courier New">% = home win / draw / away win (calibrated). '
            'xScore = single most-likely scoreline. * = the model\'s biggest disagreement with '
            f'{odds_src} &#8212; a diagnostic, not a tip; see the calibration health check below.'
            f'{rot_note}</font></p>')
    else:
        upcoming_section = ""

    # Live value bets: priced upcoming fixtures where the model sees +EV.
    from wcpredictor.betting import devig, ev_per_unit
    from wcpredictor.calibration import blend

    gamma, ll_raw, ll_cal = calib
    priced = [r for r in upcoming if r["odds"] is not None]
    # Raw (unfiltered) +EV count, for the "before/after the filter" contrast.
    n_raw = sum(1 for r in priced
                if max(ev_per_unit(r["p"][k], r["odds"][k]) for k in range(3)) > 0)
    # Filtered "value": edge is already computed on the market-shrunk model in
    # _upcoming; here we keep only those clearing the minimum-edge bar.
    value = sorted([r for r in priced if r["edge"] > MIN_EDGE], key=lambda r: -r["edge"])
    n_priced, n_value = len(priced), len(value)
    calib_note = (
        f'<b><font color="#CC0000">&#128680; HEALTH CHECK:</font></b> the raw model "finds value" on '
        f'<b>{n_raw} of {n_priced}</b> priced games &mdash; a classic over-eager-model tell, not a goldmine. '
        f'The honest filter (shrink <b>{int(MARKET_SHRINK*100)}%</b> toward the market, then demand a '
        f'<b>&ge;{int(MIN_EDGE*100)}%</b> edge) cuts that to <b>{n_value}</b>: the biggest, most defensible '
        f'disagreements. The calibration fixes are in &mdash; spread compression removed the old elite '
        f'over-confidence (replay log-loss <b>{summary.log_loss:.3f}</b>), tournament uncertainty pulled the '
        f'outright favourite from ~31% to <b>{fav_model*100:.0f}%</b>, and a temperature now fits '
        f'&gamma;={gamma:.2f} (&asymp;1). But it is still an Elo-only estimator, so treat the list as '
        f'<i>"where it disagrees with the bookies"</i>, and trust only the realized scoreboard below over '
        f'a real, larger sample.')
    val_rows = ""
    for r in value[:12]:
        mh, md, ma = r["mkt"]
        mkt_pct = (mh, md, ma)[r["sel"]]
        val_rows += (f'<tr bgcolor="{"#FFFFCC" if len(val_rows) % 2 else "#FFFFFF"}">'
                     f'<td>&nbsp;{r["dt"]:%a %d %b}</td>'
                     f'<td>&nbsp;{name[r["h"]]} v {name[r["a"]]}</td>'
                     f'<td align="center"><b>{_sel_label(r, r["sel"])}</b></td>'
                     f'<td align="center">{r["odds"][r["sel"]]:.2f}</td>'
                     f'<td align="center">{r["p"][r["sel"]]*100:.0f}%</td>'
                     f'<td align="center">{mkt_pct*100:.0f}%</td>'
                     f'<td align="center"><b><font color="#008000">+{r["edge"]*100:.0f}%</font></b></td></tr>')
    if not val_rows:
        val_rows = ('<tr bgcolor="#FFFFFF"><td colspan="7">&nbsp;After the filter, no bet clears the '
                    f'&ge;{int(MIN_EDGE*100)}% edge bar across {len(priced)} priced fixtures &#8212; the '
                    'model broadly agrees with the bookies.</td></tr>')

    if bet is None:
        bet_html = ('<tr bgcolor="#FFFFFF"><td colspan="2">&nbsp;No <i>priced</i> game has finished yet. '
                    'Realized profit/loss will appear here as the fixtures above are played and results '
                    'recorded.</td></tr>')
        bet_verdict = ("No priced game has settled yet, so there is nothing to bank &#8212; the real test starts "
                       "once priced fixtures finish. The structural calibration fixes are in (the elite "
                       "over-confidence and the silly top-heavy outright are gone), but the model still "
                       "<i>disagrees</i> with the bookies on most games, and history says that is mostly model "
                       "error, not value. Watch this space for realized profit/loss &#8212; and only believe it "
                       "with positive closing-line value over a real, larger sample.")
    else:
        tie = abs(bet.model_log_loss - bet.market_log_loss) < 0.005
        cmp = "matches" if tie else ("beats" if bet.model_log_loss < bet.market_log_loss else "trails")
        bet_html = (
            f'<tr bgcolor="#FFFFFF"><td>&nbsp;Settled matches with odds</td><td align="center">{bet.n_matches}</td></tr>'
            f'<tr bgcolor="#FFFFCC"><td>&nbsp;Avg bookmaker margin (the vig)</td><td align="center">{bet.avg_overround*100:.1f}%</td></tr>'
            f'<tr bgcolor="#FFFFFF"><td>&nbsp;Model vs market log-loss</td><td align="center">{bet.model_log_loss:.4f} vs {bet.market_log_loss:.4f}</td></tr>'
            f'<tr bgcolor="#FFFFCC"><td>&nbsp;+EV bets the model would place</td><td align="center">{bet.n_bets}</td></tr>'
            f'<tr bgcolor="#FFFFFF"><td>&nbsp;Flat stake (1u/bet) P/L</td><td align="center">{bet.flat_profit:+.2f}u on {bet.flat_staked:.0f}u ({bet.flat_roi*100:+.1f}%)</td></tr>'
            f'<tr bgcolor="#FFFFCC"><td>&nbsp;&frac14;-Kelly bankroll</td><td align="center">{bet.kelly_start:.0f} &rarr; {bet.kelly_end:.2f} ({bet.kelly_growth*100:+.1f}%)</td></tr>'
        )
        if bet.n_bets == 0:
            bet_verdict = (f"The model {cmp} this market on accuracy, but the {bet.avg_overround*100:.1f}% "
                           f"margin leaves <b>zero +EV bets</b>. Being as good as the market isn't enough "
                           f"&#8212; you have to be <i>better</i>.")
        elif bet.flat_profit > 0:
            bet_verdict = (f"Flagged {bet.n_bets} +EV bets for a notional <b>{bet.flat_roi*100:+.1f}%</b> "
                           f"&#8212; but on {bet.n_matches} games that is mostly luck. Believe it only with "
                           f"positive closing-line value over a real, larger sample.")
        else:
            bet_verdict = (f"Flagged {bet.n_bets} +EV bets and <b>lost {bet.flat_roi*100:+.1f}%</b> &#8212; "
                           f"the usual fate of betting into the margin.")

    acc = (f'Log-loss <b>{summary.log_loss:.3f}</b> vs {summary.baseline_log_loss:.3f} baseline '
           f'&middot; top-pick <b>{summary.hit_rate*100:.0f}%</b> &middot; '
           f'skill <b>{summary.skill*100:+.0f}%</b> over {summary.n} games') if summary else "No games yet."

    # Closing-line-value scoreboard: the honest verdict on edge. Empty until odds
    # snapshots accumulate an open->close pair on a settled game; it fills itself in.
    cbets = clv_data["bets"]
    if not cbets:
        clv_html = (
            '<p>&#8987; <b>Accumulating.</b> Closing-line value needs at least two odds '
            f'snapshots (an opening and a closing price) on a settled game. So far: '
            f'<b>{clv_data["n_snapshots"]}</b> snapshot(s) since {clv_data["first"] or "&ndash;"}, '
            f'<b>{clv_data["n_tracked"]}</b> fixtures tracked, <b>{clv_data["n_pairs"]}</b> with an '
            'open+close pair. The live updater snapshots the closing line just before each kick-off, '
            'so this lights up on its own as games are priced and played &#8212; <i>beating the '
            'closing line is the only honest proof of edge</i>.</p>')
    else:
        n = len(cbets)
        beat = sum(1 for b in cbets if b["clv"] > 0)
        mean_clv = sum(b["clv"] for b in cbets) / n
        pl = sum(b["pl"] for b in cbets)
        crows = "".join(
            f'<tr bgcolor="{"#FFFFCC" if i % 2 else "#FFFFFF"}">'
            f'<td>&nbsp;{name.get(b["fix"][1], b["fix"][1])} v {name.get(b["fix"][2], b["fix"][2])}</td>'
            f'<td align="center">{("Home","Draw","Away")[b["sel"]]}</td>'
            f'<td align="center">{b["open"]:.2f}</td><td align="center">{b["close"]:.2f}</td>'
            f'<td align="center"><b><font color="{"#008000" if b["clv"]>0 else "#CC0000"}">'
            f'{b["clv"]*100:+.0f}%</font></b></td>'
            f'<td align="center">{"WON" if b["won"] else "lost"}</td></tr>'
            for i, b in enumerate(cbets))
        clv_html = (
            f'<p>Over <b>{n}</b> settled bet(s) the model would have placed, it beat the closing '
            f'line <b>{beat}/{n}</b> times (mean CLV <b>{mean_clv*100:+.1f}%</b>), for a realized '
            f'<b>{pl:+.2f}u</b>. Beating the close matters more than the P/L on a small sample &#8212; '
            'it is the part that does not wash out as luck.</p>'
            '<table border="2" cellpadding="3" cellspacing="0" bgcolor="#FFFFFF" width="80%">'
            '<tr bgcolor="#4B0082"><th align="left"><font color="#FFFF00">&nbsp;Match</font></th>'
            '<th><font color="#FFFF00">Bet</font></th><th><font color="#FFFF00">Took</font></th>'
            '<th><font color="#FFFF00">Closed</font></th><th><font color="#FFFF00">CLV</font></th>'
            '<th><font color="#FFFF00">Result</font></th></tr>'
            f'{crows}</table>')

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>:::: WORLD CUP 2026 PREDICT-O-MATIC 3000 ::::</title>
<link rel="icon" href="{fav}">
<link rel="shortcut icon" href="{fav}">
<style>
  body {{
    background-color:#008080;
    background-image:
      radial-gradient(white 1px, transparent 1px),
      radial-gradient(white 1px, transparent 1px),
      repeating-linear-gradient(45deg,#007a7a,#007a7a 14px,#008b8b 14px,#008b8b 28px);
    background-size: 60px 60px, 60px 60px, auto;
    background-position: 0 0, 30px 30px, 0 0;
    color:#000000; font-family:"Comic Sans MS","Trebuchet MS",cursive;
    cursor: url("{cur}") 14 14, auto;
  }}
  a, button {{ cursor: url("{cur}") 14 14, pointer; }}
  .boot {{ position:fixed; left:0; top:0; pointer-events:none; z-index:9999; user-select:none; }}
  a {{ color:#0000EE; font-weight:bold; }}
  .blink {{ animation: blink 1s steps(2,start) infinite; }}
  @keyframes blink {{ to {{ visibility:hidden; }} }}
  .rainbow {{ animation: hue 4s linear infinite; }}
  @keyframes hue {{ from {{ filter:hue-rotate(0deg); }} to {{ filter:hue-rotate(360deg); }} }}
  h1,h2 {{ font-family:"Impact","Arial Black",sans-serif; }}
  .plaque {{ background:#C0C0C0; border:4px ridge #FFFFFF; padding:6px; }}
  .panel {{ border:4px outset #C0C0C0; background:#D4D0C8; padding:8px; }}
  .rbar {{ height:7px; background:linear-gradient(to right,red,orange,yellow,green,blue,violet); }}
  .cbar {{ height:16px; background:repeating-linear-gradient(45deg,#000 0 12px,#FFD400 12px 24px); }}
  .new {{ color:#FF0000; font-family:"Impact"; }}
  button {{ border:3px outset #C0C0C0; background:#C0C0C0; font-family:"MS Sans Serif",sans-serif; }}
</style></head>
<body bgcolor="#008080" text="#000000" link="#0000EE" vlink="#551A8B">
<center>
<table border="2" cellpadding="4" cellspacing="0" width="90%" bgcolor="#000000">
<tr><td align="center"><font face="Courier New" color="#00FF00" size="2">
&laquo;&laquo; THE SPORTSBALL PREDICT-O-MATIC WEBRING &raquo;&raquo;<br>
&#9664; PREV &nbsp;|&nbsp; <b>[ &#9917; World Cup ]</b> &nbsp;|&nbsp; <a href="f1.html">&#127937; Formula 1</a> &nbsp;|&nbsp; NEXT &#9654;
</font></td></tr></table>
<marquee behavior="alternate" scrollamount="6"><font size="6" color="#FFFF00" face="Impact">
&#9917; WORLD CUP 2026 PREDICT-O-MATIC 3000 &#9917;</font></marquee>

<table border="6" cellpadding="10" bgcolor="#000080" width="90%"><tr><td align="center">
<font color="#FFFF00" size="6" face="Impact"><b>~ OFFICIAL PREDICTIONS HQ ~</b></font><br>
<font color="#00FF00" face="Courier New">A self-learning football prophecy machine</font>
</td></tr></table>
<p><font face="Courier New" size="2">Last updated: <b>{updated}</b> &nbsp;|&nbsp;
<span class="blink"><font color="#FF0000">&#9679; LIVE</font></span> &nbsp;|&nbsp;
<span class="new blink">&#9733; NEW! &#9733;</span> &nbsp;|&nbsp;
&#128266; Turn your speakers on! <i>now playing: Three Lions (Football's Coming Home)</i></font></p>
<audio id="song" src="three-lions.mp3" autoplay loop preload="auto"></audio>
<p><button id="playbtn" type="button"><b>&#9654; PLAY ANTHEM</b></button>
&nbsp;<font size="1" face="Courier New">(starts on its own &#8212; click if your browser blocks autoplay)</font></p>
<div class="rbar rainbow"></div>

<table cellpadding="12"><tr valign="top"><td width="45%">
<h2><font color="#FFFFFF">&#127942; WHO WILL WIN IT?</font></h2>
<table border="2" cellpadding="3" cellspacing="0" bgcolor="#FFFFFF" width="100%">
<tr bgcolor="#000080"><th><font color="#FFFF00">#</font></th><th align="left"><font color="#FFFF00">&nbsp;Team</font></th>
<th><font color="#FFFF00">Elo</font></th><th><font color="#FFFF00">Model</font></th><th><font color="#FFFF00">Bookies</font></th></tr>
{champ_rows}
</table>
<p><font size="1" face="Courier New">{champ_caption}</font></p>
</td><td width="55%">
<h2><font color="#FFFFFF">&#128176; THE SWEEPSTAKE LEAGUE</font></h2>
<table border="2" cellpadding="3" cellspacing="0" bgcolor="#FFFFFF" width="100%">
<tr bgcolor="#800000"><th><font color="#FFFF00">Pos</font></th><th align="left"><font color="#FFFF00">&nbsp;Player</font></th>
<th><font color="#FFFF00">Win%</font></th><th align="left"><font color="#FFFF00">&nbsp;Best team</font></th></tr>
{sweep_rows}
</table>
<p class="plaque"><font size="2">&#128202; MODEL REPORT CARD<br>{acc}</font></p>
</td></tr></table>

<h2><font color="#FFFFFF">&#9917; GROUP STAGE &#8212; WHO'S GOING THROUGH?</font></h2>
<table cellpadding="6"><tr>{groups_html}</tr></table>

{bracket_html}
{upcoming_section}

<h2><font color="#FFFFFF">&#127922; CAN WE BEAT THE BOOKIES?</font></h2>
<p><b>LIVE &#8212; where the model most disagrees with the real bookies</b>
({odds_src}). A "value" line is one where the model's
probability beats the odds even after the margin &#8212; but read the health check:
these are overwhelmingly the model being wrong, not free money.</p>
<table border="2" cellpadding="3" cellspacing="0" bgcolor="#FFFFFF" width="90%">
<tr bgcolor="#006400"><th align="left"><font color="#FFFF00">&nbsp;When</font></th>
<th align="left"><font color="#FFFF00">&nbsp;Match</font></th>
<th><font color="#FFFF00">Bet</font></th><th><font color="#FFFF00">Odds</font></th>
<th><font color="#FFFF00">Model</font></th><th><font color="#FFFF00">Market</font></th>
<th><font color="#FFFF00">Edge</font></th></tr>
{val_rows}
</table>
<p>{calib_note}</p>
<p><font size="1" face="Courier New">&#9888; "Edge" is the model's <i>claimed</i> advantage after shrinking toward the market, not a guarantee &#8212; it is only real if it holds up over many games with positive closing-line value. Showing the top 12 by edge. Stake responsibly; this is for fun, not financial advice.</font></p>

<h3><font color="#FFFFFF">&#127942; Softer market: outright winner value</font></h3>
{outright_html}

<h3><font color="#FFFFFF">&#128202; Realized scoreboard (settled priced games)</font></h3>
<table border="2" cellpadding="3" cellspacing="0" bgcolor="#FFFFFF" width="75%">
<tr bgcolor="#000080"><th align="left"><font color="#FFFF00">&nbsp;The bookie battle</font></th>
<th><font color="#FFFF00">Result</font></th></tr>
{bet_html}
</table>
<p><b><font color="#CC0000">VERDICT:</font></b> {bet_verdict}</p>

<h3><font color="#FFFFFF">&#127919; CLOSING-LINE VALUE &#8212; the real edge test</font></h3>
<p><font size="1" face="Courier New">For every bet the model would place, we compare the price it would have
<b>taken</b> (the opening odds snapshot) with the <b>closing</b> line (last snapshot before kickoff). Beating the
close &#8212; positive CLV &#8212; is the one signal that genuinely separates skill from luck, so this is the
scoreboard that matters.</font></p>
{clv_html}

<h2><font color="#FFFFFF">&#128221; LATEST PREDICTIONS vs REALITY</font></h2>
<table border="2" cellpadding="3" cellspacing="0" bgcolor="#FFFFFF" width="60%">
<tr bgcolor="#000080"><th align="left"><font color="#FFFF00">&nbsp;Match</font></th>
<th><font color="#FFFF00">Pick</font></th><th><font color="#FFFF00">Result</font></th></tr>
{pred_rows}
</table>

<br><div class="cbar"></div>
<p><font face="Impact" size="4" color="#FFFF00">&#128679; THIS SITE IS UNDER CONSTRUCTION &#128679;</font></p>
<div class="cbar"></div><br>

<table border="3" cellpadding="6" bgcolor="#000000"><tr><td align="center">
<font color="#00FF00" face="Courier New">YOU ARE VISITOR NUMBER</font><br>
<font color="#FFFF00" face="Courier New" size="5"><b><span id="hitcount">{_hits():07d}</span></b></font>
</td></tr></table>

<p class="panel"><font size="2" face="MS Sans Serif">
&#128279; <b>THE WORLD CUP WEBRING</b> &#128279;<br>
[ <a href="#">&#171; Prev</a> | <a href="#">Random</a> | <a href="#">List Sites</a> | <a href="#">Next &#187;</a> ]
</font></p>

<p><font size="2" face="Courier New">
Best viewed in <b>Netscape Navigator 4.0</b> at 800x600 &#8212;
<a href="#">Sign our guestbook!</a> &#8212; <a href="#">&#9993; Email the webmaster</a><br>
<img alt="[Netscape Now!]" src="data:image/gif;base64,R0lGODlhEAAQAIAAAAAAAP///yH5BAEAAAEALAAAAAAQABAAAAIgjI+py+0Po5y02ouz3rz7D4biSJbmiabqyrbuC8fyTBcAOw==">
Powered by wcpredictor &#8212; not financial advice!
</font></p>
<p><font size="1">&copy; 2026 Predict-O-Matic 3000. Made with &#10084; and a Monte Carlo simulation. webmaster@predictomatic.geocities.com</font></p>
</center>
<script>
/* Real visitor counter: increments a free no-signup counter on each load and
   shows the true total. Falls back to the offline number if the service is down. */
(function () {{
  var el = document.getElementById("hitcount");
  if (!el) return;
  fetch("https://abacus.jasoncameron.dev/hit/wc2026-predictomatic/home")
    .then(function (r) {{ return r.json(); }})
    .then(function (d) {{
      var n = d && (d.value !== undefined ? d.value : d.count);
      if (typeof n === "number") el.textContent = String(n).padStart(7, "0");
    }})
    .catch(function () {{ /* keep the fallback number */ }});
}})();
</script>
<script>
/* Autoplay the anthem, with a manual play/pause fallback for strict browsers. */
(function () {{
  var a = document.getElementById("song"); if (!a) return; a.volume = 0.7;
  var btn = document.getElementById("playbtn");
  function upd() {{ if (btn) btn.innerHTML = a.paused ? "<b>&#9654; PLAY ANTHEM</b>" : "<b>&#9208; PAUSE ANTHEM</b>"; }}
  function start() {{ var p = a.play(); if (p && p.catch) p.catch(function () {{}}); }}
  if (btn) btn.addEventListener("click", function () {{ if (a.paused) a.play(); else a.pause(); }});
  a.addEventListener("play", upd); a.addEventListener("pause", upd);
  start();
  var go = function () {{ start(); document.removeEventListener("click", go); document.removeEventListener("keydown", go); }};
  document.addEventListener("click", go); document.addEventListener("keydown", go);
}})();
</script>
<script>
/* Football-boot cursor trail (the football itself is the cursor). */
(function () {{
  var N = 10, dots = [], mx = window.innerWidth / 2, my = window.innerHeight / 2;
  for (var i = 0; i < N; i++) {{
    var s = document.createElement("span");
    s.className = "boot"; s.textContent = "\\uD83D\\uDC5F";
    s.style.fontSize = (20 - i * 1.3) + "px";
    s.style.opacity = (1 - i / (N + 2));
    document.body.appendChild(s);
    dots.push({{ el: s, x: mx, y: my }});
  }}
  document.addEventListener("mousemove", function (e) {{ mx = e.clientX; my = e.clientY; }});
  (function loop() {{
    var px = mx, py = my;
    for (var i = 0; i < dots.length; i++) {{
      var d = dots[i];
      d.x += (px - d.x) * 0.35; d.y += (py - d.y) * 0.35;
      d.el.style.transform = "translate(" + (d.x - 8) + "px," + (d.y - 8) + "px)";
      px = d.x; py = d.y;
    }}
    requestAnimationFrame(loop);
  }})();
}})();
</script>
</body></html>"""


if __name__ == "__main__":
    main()
