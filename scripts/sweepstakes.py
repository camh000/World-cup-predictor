"""Compute sweepstake win probabilities and write them to a markdown file.

Win chance = combined title probability of a person's team(s) (exactly one team
wins the cup, so the events are mutually exclusive and the probabilities add).

Run after importing the latest results to refresh the odds:
    wcpredict import-results --file <official schedule>
    wcpredict replay                     # update ratings from played games
    python scripts/sweepstakes.py
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from pathlib import Path

from wcpredictor.config import Params, Paths
from wcpredictor.data_io import _norm_name, build_name_index, read_teams
from wcpredictor.ratings import RatingStore
from wcpredictor.simulate import run_simulation

OUT = Path("sweepstake_odds.md")
N_SIMS = 50000
SEED = 7

# --- the two sweepstakes -----------------------------------------------------
FRIENDS = {
    "Jake": ["Haiti", "Norway", "Iraq"],
    "Jono": ["South Africa", "Bosnia", "Jordan"],
    "Joey": ["Portugal", "Switzerland", "Cape Verde"],
    "Willis": ["Spain", "Qatar", "Ivory Coast"],
    "Cam": ["USA", "Colombia", "Curacao"],
    "Tegan": ["Senegal", "Tunisia", "Japan"],
    "Max": ["France", "Panama", "DR Congo"],
    "Jack": ["Belgium", "Austria", "Morocco"],
    "Oliver": ["Canada", "England", "Iran"],
    "Caleb": ["Saudi Arabia", "Argentina", "Egypt"],
    "Bailey": ["Uzbekistan", "Netherlands", "Paraguay"],
    "Jeza": ["Uruguay", "Mexico", "South Korea"],
    "Liam": ["Ecuador", "Germany", "Australia"],
    "Callum": ["Sweden", "Czech Republic", "Brazil"],
    "Matt B": ["Scotland", "Algeria", "New Zealand"],
    "Matt N": ["Croatia", "Ghana", "Turkey"],
}

# team -> (nominee 1, nominee 2)
WORK = [
    ("Mexico", "Emma Tissington", "Emma Tissington"),
    ("South Africa", "Carly Turner", "Rachael Teebay"),
    ("South Korea", "Adam Tevendale", "Dom Audi"),
    ("Czechia", "Brittany Bedford", "Khalid Edwards"),
    ("Canada", "Jamie Turner", "Amirah Saoti"),
    ("Bosnia and Herzegovina", "James Wenninger", "Angela Hughes"),
    ("Qatar", "Sue Neville", "Zoe Jones"),
    ("Switzerland", "Josh Barker", "Matt Ball"),
    ("Brazil", "Karl Hantke", "Nicola Tiffin"),
    ("Morocco", "Laura Fletcher", "Beckie Waistnidge"),
    ("Haiti", "Karin Swafenberg", "Cameron Sinclair"),
    ("Scotland", "Andy Barrow", "Josh Barker"),
    ("United States", "Dan Hickman", "Ryan Kerr"),
    ("Paraguay", "Josh Confue", "Steve Dennis"),
    ("Australia", "Tony Barker", "Simon Parkes"),
    ("Turkey", "James Hewitt", "Andy Barrow"),
    ("Germany", "Andrew Henry", "Chris Woodcock"),
    ("Curacao", "John Wright", "Alice Kay"),
    ("Ivory Coast", "Toni Royle", "Wayne Roome"),
    ("Ecuador", "Suzanne Wilson", "Sylwia Rybak"),
    ("Netherlands", "Lee Hughes", "Mark Lambert"),
    ("Japan", "Lynne Smith", "Sheril Tucker"),
    ("Sweden", "Steve Graydon", "Sue Miles"),
    ("Tunisia", "Jack Roebuck", "Cameron Hall"),
    ("Belgium", "Will Evans", "Trav White"),
    ("Egypt", "Emily Martin", "Oliver Fairhurst"),
    ("Iran", "Rushali Kumar", "Zoe Jones"),
    ("New Zealand", "Andy Barrow", "Aisling Goodbody"),
    ("Spain", "Angela Hughes", "Tim Adams"),
    ("Cape Verde", "Jamie Turner", "James Widdowson"),
    ("Saudi Arabia", "Lydia Mayes", "Craig Smith"),
    ("Uruguay", "David Thompson", "Jaydene Venus"),
    ("France", "Jason Hyde", "Isaac Woffenden"),
    ("Senegal", "David Thompson", "Alice Newton"),
    ("Iraq", "Devon Kelsall", "Katie Marsden"),
    ("Norway", "Mel Sinclair", "Chris Hayward"),
    ("Argentina", "Emma Wakefield", "Lauren Cox"),
    ("Algeria", "Alice Newton", "Andy Barrow"),
    ("Austria", "Andrew Woodcock", "Emma Tissington"),
    ("Jordan", "Helen Bonar", "Kate Cooper"),
    ("Portugal", "Ella Thomas", "Rachael Lomax"),
    ("DR Congo", "Laura Evans", "Luke Carr"),
    ("Uzbekistan", "David Fitzgerald", "Ryan Kerr"),
    ("Colombia", "Stuart Hyde", "Ryan Gregory"),
    ("England", "Sophie Jackson", "Helena Young"),
    ("Croatia", "Man-Tong Li", "Kara Kerkhoff"),
    ("Ghana", "Andy Waite", "Ryan Gregory"),
    ("Panama", "Lauren Cox", "Zoe Jones"),
]

EXTRA_ALIASES = {"bosnia": "BIH", "czech republic": "CZE", "panama": "PAN"}


def abbreviate_surnames(full_names) -> dict:
    """Map "First Last" -> "First L", extending the surname initial (L, La, ...)
    only as far as needed to keep every abbreviation unique."""
    names = list(dict.fromkeys(full_names))

    def split(n):
        parts = n.split()
        return parts[0], (parts[-1] if len(parts) > 1 else "")

    out = {}
    for n in names:
        first, sur = split(n)
        k = 1
        while True:
            ab = f"{first} {sur[:k]}".strip()
            clash = any(
                m != n and split(m)[0] == first and split(m)[1][:k] == sur[:k]
                for m in names
            )
            if not clash or k >= len(sur):
                out[n] = ab
                break
            k += 1
    return out


def main() -> None:
    paths = Paths()
    teams = read_teams(paths.teams_csv)
    params = Params.load(paths.params_json)
    ratings = RatingStore.load(paths.ratings_json)
    df = run_simulation(teams, params, ratings, n_sims=N_SIMS, seed=SEED)
    pchamp = {r.team_id: r.p_champion for r in df.itertuples()}

    idx = build_name_index(teams)
    for alias, tid in EXTRA_ALIASES.items():
        idx[_norm_name(alias)] = tid

    def cid(name: str) -> str:
        tid = idx.get(_norm_name(name))
        if not tid:
            raise SystemExit(f"unmapped team name: {name!r}")
        return tid

    lines = [
        "# World Cup 2026 Sweepstake Odds",
        "",
        f"_Model estimate as of {date.today().isoformat()} "
        f"({N_SIMS:,} simulations from the current post-matchday ratings)._",
        "",
        "Win chance = combined title probability of a person's team(s).",
        "",
        "## Friend group",
        "",
        "| # | Person | Win chance | Teams (best) |",
        "|---|--------|-----------:|--------------|",
    ]
    rows = []
    for person, tt in FRIENDS.items():
        total = sum(pchamp[cid(t)] for t in tt)
        best = max(tt, key=lambda t: pchamp[cid(t)])
        rows.append((person, total, tt, best))
    for i, (person, total, tt, best) in enumerate(sorted(rows, key=lambda r: -r[1]), 1):
        lines.append(f"| {i} | {person} | {total*100:.1f}% | "
                     f"{', '.join(tt)} (best: {best} {pchamp[cid(best)]*100:.1f}%) |")

    person_teams = defaultdict(set)
    for team, n1, n2 in WORK:
        person_teams[n1].add(team)
        person_teams[n2].add(team)

    lines += [
        "",
        f"## Work sweepstake ({len(person_teams)} people)",
        "",
        "| # | Person | Win chance | Team(s) |",
        "|---|--------|-----------:|---------|",
    ]
    short = abbreviate_surnames(person_teams.keys())
    wr = []
    for person, tt in person_teams.items():
        total = sum(pchamp[cid(t)] for t in tt)
        ordered = sorted(tt, key=lambda t: -pchamp[cid(t)])
        wr.append((short[person], total, ordered))
    for i, (person, total, tt) in enumerate(sorted(wr, key=lambda r: (-r[1], r[0])), 1):
        lines.append(f"| {i} | {person} | {total*100:.2f}% | {', '.join(tt)} |")

    fav = df.iloc[0]
    lines += ["", f"_Tournament favourite: {fav['team']} ({fav['p_champion']*100:.1f}%)._", ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT} ({len(rows)} friends, {len(person_teams)} work entrants)")


if __name__ == "__main__":
    main()
