"""Generate a gloriously early-2000s F1 page (f1.html) — sister site to the
World Cup predictor in the webring. Drivers'/constructors' championship odds and
next-race podium probabilities from wcpredictor.f1.
"""

from __future__ import annotations

import urllib.parse
from datetime import datetime
from pathlib import Path

from wcpredictor.f1 import (
    constructor_of, current_grid, load_races, next_race_probs, rate_drivers,
    simulate_championship, standings,
)

ROOT = Path(__file__).resolve().parents[1]
F1DIR = ROOT / "data" / "f1"
OUT = ROOT / "f1.html"

TOTAL_RACES = 24      # 2026 calendar length (the dataset has no 2026 calendar file)
N_SIMS = 20000
SCALE = 110.0
DNF_PROB = 0.12


def _flag_favicon() -> str:
    """An 8x8 checkered-flag SVG data-URI favicon."""
    sq = "".join(
        f'<rect x="{x}" y="{y}" width="1" height="1" fill="#000"/>'
        for y in range(8) for x in range(8) if (x + y) % 2 == 0)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8" '
           f'shape-rendering="crispEdges"><rect width="8" height="8" fill="#fff"/>{sq}</svg>')
    return "data:image/svg+xml," + urllib.parse.quote(svg)


def _webring(active: str) -> str:
    """Shared webring nav bar; ``active`` is 'football' or 'f1'."""
    f = "<b>[ &#9917; World Cup ]</b>" if active == "football" else '<a href="index.html">&#9917; World Cup</a>'
    o = "<b>[ &#127937; Formula 1 ]</b>" if active == "f1" else '<a href="f1.html">&#127937; Formula 1</a>'
    return (
        '<table border="2" cellpadding="4" cellspacing="0" width="100%" bgcolor="#000000">'
        '<tr><td align="center"><font face="Courier New" color="#00FF00" size="2">'
        '&laquo;&laquo; THE SPORTSBALL PREDICT-O-MATIC WEBRING &raquo;&raquo;<br>'
        f'&#9664; PREV &nbsp;|&nbsp; {f} &nbsp;|&nbsp; {o} &nbsp;|&nbsp; NEXT &#9654;'
        '</font></td></tr></table>')


def _render(rows, con_rows, race_rows, completed, updated) -> str:
    fav = _flag_favicon()
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>:::: F1 2026 PREDICT-O-MATIC :::: GRID OF DESTINY ::::</title>
<link rel="icon" href="{fav}"><link rel="shortcut icon" href="{fav}">
<style>
  body {{ background:#1a1a1a; color:#FFFFFF; font-family:"Comic Sans MS","Trebuchet MS",cursive;
         background-image:repeating-linear-gradient(45deg,#1a1a1a 0 20px,#222 20px 40px); }}
  h1,h2 {{ text-align:center; }}
  table {{ margin:0 auto; }}
  a {{ color:#FF1801; }}
  .wrap {{ max-width:860px; margin:0 auto; }}
  marquee {{ color:#FFD700; font-weight:bold; }}
</style></head>
<body>
<div class="wrap">
{_webring('f1')}
<h1><font color="#FF1801" size="6">&#127937; FORMULA 1 &#8212; 2026 PREDICT-O-MATIC</font></h1>
<marquee>&#9888; Lights out and away we go! Powered by a pairwise-Elo + Plackett-Luce Monte Carlo &#9888;</marquee>
<p align="center"><font size="1">Last updated {updated} &middot; {completed} of {TOTAL_RACES} races run</font></p>

<h2><font color="#FFD700">&#127942; DRIVERS' CHAMPIONSHIP</font></h2>
<table border="2" cellpadding="3" cellspacing="0" bgcolor="#000000" width="90%">
<tr bgcolor="#FF1801"><th><font color="#FFF">#</font></th><th align="left"><font color="#FFF">&nbsp;Driver</font></th>
<th align="left"><font color="#FFF">&nbsp;Constructor</font></th><th><font color="#FFF">Pts</font></th>
<th><font color="#FFF">Title %</font></th><th><font color="#FFF">Proj. pts</font></th></tr>
{rows}
</table>

<h2><font color="#FFD700">&#127981; CONSTRUCTORS' CHAMPIONSHIP</font></h2>
<table border="2" cellpadding="3" cellspacing="0" bgcolor="#000000" width="70%">
<tr bgcolor="#FF1801"><th><font color="#FFF">#</font></th><th align="left"><font color="#FFF">&nbsp;Constructor</font></th>
<th><font color="#FFF">Pts</font></th><th><font color="#FFF">Title %</font></th></tr>
{con_rows}
</table>

<h2><font color="#FFD700">&#127937; NEXT GRAND PRIX &#8212; PODIUM ODDS</font></h2>
<table border="2" cellpadding="3" cellspacing="0" bgcolor="#000000" width="80%">
<tr bgcolor="#FF1801"><th align="left"><font color="#FFF">&nbsp;Driver</font></th><th><font color="#FFF">Win</font></th>
<th><font color="#FFF">Podium</font></th><th><font color="#FFF">Points</font></th></tr>
{race_rows}
</table>
<p><font size="1" face="Courier New">Driver ratings are a pairwise Elo over 2025+2026 finishing orders (the rating
absorbs car pace). Races are simulated Plackett-Luce style (Gumbel-perturbed ratings), {N_SIMS} runs, with a
{DNF_PROB*100:.0f}% per-driver DNF chance. Remaining season assumed {TOTAL_RACES} races; sprints and
track-specific effects not modelled. Data: toUpperCase78/formula1-datasets. Just for fun.</font></p>
{_webring('f1')}
</div>
</body></html>"""


def _sprint_rows(path):
    """(driver, team, points) rows from a sprint-results CSV (for standings)."""
    import csv
    out = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                pts = float(r["Points"])
            except (ValueError, KeyError):
                pts = 0.0
            out.append((r["Driver"].strip(), r["Team"].strip(), pts))
    return out


def main() -> None:
    r25 = load_races(F1DIR / "race_results_2025.csv")
    r26 = load_races(F1DIR / "race_results_2026.csv")
    if not r26:
        raise SystemExit("no 2026 F1 race data; run scripts/fetch_f1.py first")
    elo = rate_drivers([r25, r26])
    drivers, team_of = current_grid(r26)
    dp, cp = standings([r26], _sprint_rows(F1DIR / "sprint_results_2026.csv"))
    completed = len(r26)
    remaining = max(0, TOTAL_RACES - completed)

    dt_odds, ct_odds, proj = simulate_championship(
        drivers, team_of, elo, dp, cp, remaining,
        n_sims=N_SIMS, scale=SCALE, dnf_prob=DNF_PROB)
    race = next_race_probs(drivers, elo, n_sims=N_SIMS, scale=SCALE, dnf_prob=DNF_PROB)

    rows = ""
    for i, d in enumerate(sorted(drivers, key=lambda x: -dp.get(x, 0.0))):
        rows += (f'<tr bgcolor="{"#222" if i % 2 else "#111"}">'
                 f'<td align="center">{i+1}</td><td>&nbsp;{d}</td>'
                 f'<td>&nbsp;{team_of[d]}</td>'
                 f'<td align="center">{dp.get(d,0.0):.0f}</td>'
                 f'<td align="center"><b><font color="#FFD700">{dt_odds.get(d,0.0)*100:.1f}%</font></b></td>'
                 f'<td align="center">{proj.get(d,0.0):.0f}</td></tr>')

    con_pts = {}
    for d in drivers:
        con_pts[team_of[d]] = cp.get(team_of[d], 0.0)
    con_rows = ""
    for i, t in enumerate(sorted(con_pts, key=lambda x: -con_pts[x])):
        con_rows += (f'<tr bgcolor="{"#222" if i % 2 else "#111"}">'
                     f'<td align="center">{i+1}</td><td>&nbsp;{t}</td>'
                     f'<td align="center">{con_pts[t]:.0f}</td>'
                     f'<td align="center"><b><font color="#FFD700">{ct_odds.get(t,0.0)*100:.1f}%</font></b></td></tr>')

    race_rows = ""
    for i, d in enumerate(sorted(drivers, key=lambda x: -race[x][0])[:12]):
        w, pod, pts = race[d]
        race_rows += (f'<tr bgcolor="{"#222" if i % 2 else "#111"}">'
                      f'<td>&nbsp;{d}</td>'
                      f'<td align="center">{w*100:.0f}%</td>'
                      f'<td align="center">{pod*100:.0f}%</td>'
                      f'<td align="center">{pts*100:.0f}%</td></tr>')

    updated = datetime.utcnow().strftime("%A %d %B %Y, %H:%M UTC")
    OUT.write_text(_render(rows, con_rows, race_rows, completed, updated), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
