"""Regenerate data/fox_falcon_bf.txt: every 2-player Fox vs Captain Falcon game on Battlefield.

The upstream dataset lists some games twice, once under FOX/<name>.slp and again under
FOX/batch_NN/<name>.slp, so the basename is the identity. Filenames carry the matchup, e.g.
"FOX/batch_01/19_42_02 [NEWT] Fox + [ROBN] Captain Falcon (BF).slp", and a game with more than
two characters has more than one " + ", which is how 4-player games are dropped.

    python data/select_games.py            # writes data/fox_falcon_bf.txt
    python data/select_games.py --check    # verify the committed list still matches
"""
import os, sys
from huggingface_hub import list_repo_files

REPO = "erickfm/slippi-public-dataset-v3.7"
OUT = "data/fox_falcon_bf.txt"

seen, games = set(), []
for f in sorted(list_repo_files(REPO, repo_type="dataset")):
    if not (f.startswith("FOX/") and "Captain Falcon" in f and "(BF)" in f):
        continue
    if f.count(" + ") != 1:          # two characters means one separator
        continue
    name = os.path.basename(f)
    if name not in seen:
        seen.add(name)
        games.append(f)

if "--check" in sys.argv:
    have = [l for l in open(OUT).read().split("\n") if l]
    print(f"upstream {len(games)} games, committed list {len(have)}")
    print("match" if set(games) == set(have) else "MISMATCH: upstream has changed")
else:
    open(OUT, "w").write("\n".join(games))
    print(f"wrote {OUT}: {len(games)} games")
