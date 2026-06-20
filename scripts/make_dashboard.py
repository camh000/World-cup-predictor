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
from wcpredictor.history import read_predictions, summarize
from wcpredictor.ratings import RatingStore
from wcpredictor.scenarios import qualification
from wcpredictor.simulate import run_simulation
from wcpredictor.tournament import TeamStanding

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "index.html"
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

    html = _render(df, name, base, adv, win, preds, summary, friend_rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT}")


# --------------------------------------------------------------------------- #
# Rendering (intentionally retro)
# --------------------------------------------------------------------------- #
def _render(df, name, base, adv, win, preds, summary, friend_rows) -> str:
    updated = datetime.utcnow().strftime("%A %d %B %Y, %H:%M UTC")
    fav = _favicon()

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
  }}
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
&#128266; Turn your speakers on! <i>now playing: world_cup_anthem.mid</i></font></p>
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
<font color="#FFFF00" face="Courier New" size="5"><b>{_hits():07d}</b></font>
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
</body></html>"""


if __name__ == "__main__":
    main()
