import itertools as it
import re
from pathlib import Path

import pandas as pd

## note: other sources of data were manually copied from the logs
## this only automates the two most annoying ones

ROOT = Path("/u/ianchen3/community_search/output")


def collect(path, extractor, **param_lists):
    """
    Generic driver: iterates the cartesian product of param_lists (kwargs of
    name -> list of values), builds a path from `path.format(**params)`,
    and calls extract(file_path, params) -> dict[stat, value] | None.
    Skips missing files and extraction failures. Returns a list of row
    dicts, each including all params (e.g. pass method=[...] like any other
    param, even a single-item list).
    """
    rows = []
    keys = list(param_lists.keys())
    for combo in it.product(*param_lists.values()):
        params = dict(zip(keys, combo))
        file_path = Path(path.format(**params))
        if not file_path.exists():
            continue
        result = extractor(file_path, params)
        if not result:
            continue
        for stat, value in result.items():
            rows.append({**params, "stat": stat, "value": value})
    return rows


### ===========================================================================
### extractors


def extract_mytime(path, stats):
    """Parse 'key=value' lines from mytime.py"""
    with open(path, "r") as f:
        content = f.read()
    result = {}
    for stat in stats:
        match = re.search(rf"{stat}=([^\s\n]+)", content)
        if match:
            result[stat] = match.group(1)
    return result


def extract_csk(path, b):
    """Collect wall_s for csk to handle all queries in batch"""
    with open(path, "r") as f:
        lines = [line for line in f.read().splitlines() if line.strip()]
    if len(lines) != b:
        return None
    total_ms = sum(int(line.split(",")[1]) for line in lines)
    return {"wall_s": total_ms / 1000}


def extract_querytimes(path, params):
    """Collect wall_s for ours-python-commsearch to handle all queries in batch"""
    df_file = pd.read_csv(path)
    row = df_file[df_file["index"] == "all"]
    if row.empty:
        return None
    return {"wall_s": row["wall_s"].iloc[0]}


### ===========================================================================
### training: data for core decomposition

TRAIN_NETWORKS = ["cen", "abm14"]
TRAIN_METHODS = ["ib", "gbbs", "pkc", "nk", "ucr"]
CORE_DECOMP_STATS = [
    "exit_code",
    "wall_s",
    "user_s",
    "sys_s",
    "cpu_pct",
    "peak_rss_kb",
    "peak_rss_anon_kb",
    "peak_rss_file_kb",
    "peak_rss_tree_kb",
    "peak_minflt",
    "peak_majflt",
]

train_rows = collect(
    f"{ROOT}/{{network}}/{{method}}-core-decomp/timing.txt",
    lambda path, params: extract_mytime(path, CORE_DECOMP_STATS),
    network=TRAIN_NETWORKS,
    method=TRAIN_METHODS,
)

pd.DataFrame(train_rows).to_csv("train-core-decomp.csv", index=False)
print(f"Wrote {len(train_rows)} rows to train-core-decomp.csv")


### ===========================================================================
### testing: comparing our methods versus SotA

TEST_NETWORKS = [
    "bitcoin",
    "dbpedia_link",
    "friendster",
    "livejournal",
    "microsoft_concept",
    "twitter_social",
    "wikipedia_link",
]
B = [1, 10, 100]
R = list(range(20))

# CSK: n = {1}; {network}/testing-csk/n{n}-b{b}-rep{r}/timing.log
csk_rows = collect(
    f"{ROOT}/{{network}}/testing-csk/n{{n}}-b{{b}}-rep{{r}}/timing.log",
    lambda path, params: extract_csk(path, params["b"]),
    network=TEST_NETWORKS,
    method=["csk"],
    n=[1],
    b=B,
    r=R,
)

# ParShellstruct: n = {1,5,10,20}; {network}/testing-par-shellstruct/querytimes-n{n}-b{b}-rep{r}.csv
parshell_rows = collect(
    f"{ROOT}/{{network}}/testing-par-shellstruct/querytimes-n{{n}}-b{{b}}-rep{{r}}.csv",
    extract_querytimes,
    network=TEST_NETWORKS,
    method=["par-shellstruct"],
    n=[1, 5, 10, 20],
    b=B,
    r=R,
)

# Steiner: n = {1,5,10,20}; {network}/testing-steiner/querytimes-n{n}-b{b}-rep{r}.csv
steiner_rows = collect(
    "{network}/testing-steiner/querytimes-n{n}-b{b}-rep{r}.csv",
    extract_querytimes,
    network=TEST_NETWORKS,
    method=["steiner"],
    n=[1, 5, 10, 20],
    b=B,
    r=R,
)

test_rows = csk_rows + parshell_rows + steiner_rows
pd.DataFrame(test_rows).to_csv("testing-commsearch.csv", index=False)
print(f"Wrote {len(test_rows)} rows to testing-commsearch.csv")


## data manually collected:
# - gullo shellstruct did not finish on everything
# - lbug-core-decomp did not finish on everything
# - strong-scaling of icebug
# - network statistics
# - comparison of ours-py-v-ib
# - analysis of centrality and queries
# - testing offline stages (icebug core-decomp and shellstruct)
