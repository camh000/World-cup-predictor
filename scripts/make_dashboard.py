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
from wcpredictor.data_io import read_matches, read_teams
from wcpredictor.betting import evaluate
from wcpredictor.history import read_predictions, summarize
from wcpredictor.ratings import RatingStore
from wcpredictor.scenarios import qualification
from wcpredictor.simulate import run_simulation
from wcpredictor.tournament import TeamStanding

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "index.html"   # repo root so Vercel/zero-config static serves it at "/"
N_SIMS = 20000


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


def main() -> None:
    paths = Paths()
    teams = read_teams(paths.teams_csv)
    params = Params.load(paths.params_json)
    ratings = RatingStore.load(paths.ratings_json) if paths.ratings_json.exists() \
        else RatingStore.seed(teams, {})
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

    upcoming = _upcoming(paths, teams, params, ratings, name, n=40)
    bet = _betting(paths, preds)

    html = _render(df, name, base, adv, win, preds, summary, friend_rows, upcoming, bet)
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


def _upcoming(paths, teams, params, ratings, name, n=40):
    """Rich forecasts for the next ``n`` unplayed group-stage fixtures.

    Each row is a dict with the model's probabilities and most-likely score, plus
    — where we hold bookmaker odds — the de-vigged market probabilities and the
    model's best +EV ("value") selection against that price.
    """
    import csv
    import numpy as np
    from wcpredictor.betting import devig, ev_per_unit
    from wcpredictor.data_io import HOST_TEAM_IDS, _norm_name, build_name_index
    from wcpredictor.history import forecast
    from wcpredictor.poisson import dixon_coles_matrix

    idx = build_name_index(teams)
    odds_map = _load_odds(paths)
    sched = paths.data_dir / "fifa_worldcup_2026_schedule.csv"
    rows = []
    with sched.open("r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if (r.get("Result") or "").strip():
                continue
            if r.get("Round Number", "").strip() not in {"1", "2", "3"}:
                continue
            h = idx.get(_norm_name(r.get("Home Team", "")))
            a = idx.get(_norm_name(r.get("Away Team", "")))
            if not h or not a:
                continue
            try:
                dt = datetime.strptime(r["Date"].strip(), "%d/%m/%Y %H:%M")
            except ValueError:
                continue
            rows.append((dt, h, a))
    rows.sort(key=lambda x: x[0])

    out = []
    for dt, h, a in rows[:n]:
        neutral = h not in HOST_TEAM_IDS
        probs, (lh, la) = forecast(ratings, params, h, a, neutral)
        m = dixon_coles_matrix(lh, la, params.dc_rho, params.max_goals)
        i, j = np.unravel_index(int(np.argmax(m)), m.shape)
        row = {"dt": dt, "h": h, "a": a, "p": probs, "score": (int(i), int(j)),
               "odds": None, "mkt": None, "sel": None, "edge": None}
        odds = odds_map.get((h, a))
        if odds:
            evs = [ev_per_unit(probs[k], odds[k]) for k in range(3)]
            sel = max(range(3), key=evs.__getitem__)
            row.update(odds=odds, mkt=devig(odds), sel=sel, edge=evs[sel])
        out.append(row)
    return out


def _betting(paths, preds):
    """Join settled predictions to bookmaker odds (data/odds.csv) and backtest."""
    import csv

    odds_path = paths.data_dir / "odds.csv"
    if not odds_path.exists():
        return None
    idx = {}
    with odds_path.open("r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            idx[(r["date"], r["home_team_id"], r["away_team_id"])] = (
                float(r["odds_home"]), float(r["odds_draw"]), float(r["odds_away"]))
    om = {"home": 0, "draw": 1, "away": 2}
    matches = []
    for r in preds:
        k = (r["date"], r["home_team_id"], r["away_team_id"])
        if k in idx and r["actual_outcome"] in om:
            matches.append(((float(r["p_home"]), float(r["p_draw"]), float(r["p_away"])),
                            idx[k], om[r["actual_outcome"]]))
    return evaluate(matches) if matches else None


# --------------------------------------------------------------------------- #
# Rendering (intentionally retro)
# --------------------------------------------------------------------------- #
def _render(df, name, base, adv, win, preds, summary, friend_rows, upcoming, bet) -> str:
    updated = datetime.utcnow().strftime("%A %d %B %Y, %H:%M UTC")
    fav = _favicon()
    cur = _cursor()

    champ_rows = "".join(
        f'<tr bgcolor="{"#FFFFCC" if i % 2 else "#FFFFFF"}">'
        f'<td align="center"><b>{i+1}</b></td><td>&nbsp;{r.team}</td>'
        f'<td align="center">{r.elo:.0f}</td>'
        f'<td align="center"><b>{r.p_champion*100:.1f}%</b></td></tr>'
        for i, r in enumerate(df.head(12).itertuples())
    )

    group_blocks = []
    for g in sorted(base):
        table = sorted(base[g].values(), key=lambda s: s.sort_key())
        rows = ""
        for s in table:
            a = adv[s.team_id]
            colour = "#CCFFCC" if a >= 0.9995 else "#FFCCCC" if a <= 0.0005 else "#FFFFFF"
            tag = "QUALIFIED!" if a >= 0.9995 else "OUT" if a <= 0.0005 else f"{a*100:.0f}%"
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
    for r in upcoming[:12]:
        ph, pd, pa = r["p"]
        si, sj = r["score"]
        pick = name[r["h"]] if ph >= max(pd, pa) else (name[r["a"]] if pa >= pd else "Draw")
        if r["odds"] is None:
            bet_cell = '<font color="#888888">no odds</font>'
        elif r["edge"] > 0:
            bet_cell = (f'<b><font color="#008000">&#9989; {_sel_label(r, r["sel"])} '
                        f'@{r["odds"][r["sel"]]:.2f} (+{r["edge"]*100:.0f}%)</font></b>')
        else:
            bet_cell = '<font color="#888888">no value</font>'
        up_rows += (f'<tr bgcolor="{"#FFFFCC" if len(up_rows) % 2 else "#FFFFFF"}">'
                    f'<td>&nbsp;{r["dt"]:%a %d %b %H:%M}Z</td>'
                    f'<td>&nbsp;{name[r["h"]]} v {name[r["a"]]}</td>'
                    f'<td align="center">{ph*100:.0f} / {pd*100:.0f} / {pa*100:.0f}</td>'
                    f'<td align="center"><b>{pick}</b></td>'
                    f'<td align="center">{si}-{sj}</td>'
                    f'<td align="center">{bet_cell}</td></tr>')

    # Live value bets: priced upcoming fixtures where the model sees +EV.
    priced = [r for r in upcoming if r["odds"] is not None]
    value = sorted([r for r in priced if r["edge"] > 0], key=lambda r: -r["edge"])
    n_priced, n_value = len(priced), len(value)
    frac = (n_value / n_priced) if n_priced else 0.0
    if frac >= 0.4:
        calib_note = (f'<b><font color="#CC0000">&#128680; HEALTH CHECK:</font></b> the model claims value on '
                      f'<b>{n_value} of {n_priced}</b> priced games ({frac*100:.0f}%). Against an efficient '
                      f'market that is almost certainly <b>mis-calibration, not a goldmine</b> &#8212; the model '
                      f'systematically overrates underdogs and draws (note the longshots up top). Read this as '
                      f'"the model disagrees with the bookies a lot", which usually means the model is wrong, '
                      f'not that you have found {n_value} free bets.')
    else:
        calib_note = (f'The model claims value on <b>{n_value} of {n_priced}</b> priced games &#8212; a '
                      f'selective, plausible number. Still only trust it if it shows positive closing-line '
                      f'value over a real, larger sample.')
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
        val_rows = ('<tr bgcolor="#FFFFFF"><td colspan="7">&nbsp;No +EV bets found across '
                    f'{len(priced)} priced fixtures &#8212; the model agrees with the bookies.</td></tr>')

    if bet is None:
        bet_html = ('<tr bgcolor="#FFFFFF"><td colspan="2">&nbsp;No <i>priced</i> game has finished yet. '
                    'Realized profit/loss will appear here as the fixtures above are played and results '
                    'recorded.</td></tr>')
        bet_verdict = ("The real test is live above &#8212; come back once these games are played to see "
                       "whether the model's value bets actually landed.")
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
<th><font color="#FFFF00">Elo</font></th><th><font color="#FFFF00">Win%</font></th></tr>
{champ_rows}
</table>
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

<h2><font color="#FFFFFF">&#128302; UPCOMING &#8212; NEXT MATCHES PREDICTED</font></h2>
<table border="2" cellpadding="3" cellspacing="0" bgcolor="#FFFFFF" width="75%">
<tr bgcolor="#000080"><th align="left"><font color="#FFFF00">&nbsp;Kickoff (UTC)</font></th>
<th align="left"><font color="#FFFF00">&nbsp;Match</font></th>
<th><font color="#FFFF00">Win / Draw / Win %</font></th>
<th><font color="#FFFF00">Tip</font></th><th><font color="#FFFF00">xScore</font></th>
<th><font color="#FFFF00">Value vs bookies</font></th></tr>
{up_rows}
</table>
<p><font size="1" face="Courier New">% = home win / draw / away win. xScore = single most-likely scoreline. "Value" = the model's best positive-expected-value bet against real bookmaker odds (oddschecker, 20 Jun 2026).</font></p>

<h2><font color="#FFFFFF">&#127922; CAN WE BEAT THE BOOKIES?</font></h2>
<p><b>LIVE &#8212; the bets the model would place against the real bookies right now</b>
(real oddschecker prices, 20 Jun 2026). A "value" bet is one where the model's
probability beats the odds even after the bookmaker's margin.</p>
<table border="2" cellpadding="3" cellspacing="0" bgcolor="#FFFFFF" width="90%">
<tr bgcolor="#006400"><th align="left"><font color="#FFFF00">&nbsp;When</font></th>
<th align="left"><font color="#FFFF00">&nbsp;Match</font></th>
<th><font color="#FFFF00">Bet</font></th><th><font color="#FFFF00">Odds</font></th>
<th><font color="#FFFF00">Model</font></th><th><font color="#FFFF00">Market</font></th>
<th><font color="#FFFF00">Edge</font></th></tr>
{val_rows}
</table>
<p>{calib_note}</p>
<p><font size="1" face="Courier New">&#9888; "Edge" is the model's <i>claimed</i> advantage, not a guarantee &#8212; it is only real if it holds up over many games with positive closing-line value. Showing the top 12 by edge. Stake responsibly; this is for fun, not financial advice.</font></p>

<h3><font color="#FFFFFF">&#128202; Realized scoreboard (settled priced games)</font></h3>
<table border="2" cellpadding="3" cellspacing="0" bgcolor="#FFFFFF" width="75%">
<tr bgcolor="#000080"><th align="left"><font color="#FFFF00">&nbsp;The bookie battle</font></th>
<th><font color="#FFFF00">Result</font></th></tr>
{bet_html}
</table>
<p><b><font color="#CC0000">VERDICT:</font></b> {bet_verdict}</p>

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
