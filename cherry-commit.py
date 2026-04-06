# CONFIG: all parameters centralized, change dry_run to False to execute


# CONFIG: all parameters centralized here, change dry_run to False to execute
CONFIG = {
    "repo_path": r"D:\anvo\yaaat-dev",   # absolute path to repo root
    "omit_file": "OMIT.txt",              # file listing paths to exclude, one per line
    "source_branch": "dev-local",         # branch containing all commits
    "target_branch": "dev-public",        # branch to cherry-pick onto
    "dry_run": True,                      # True = print only, False = execute cherry-pick
}

import subprocess
from pathlib import Path

def run(cmd, cwd):
    # execute shell command in given directory, return stdout as string
    return subprocess.check_output(cmd, cwd=cwd, text=True, shell=True).strip()

def is_omitted(path, omit_paths):
    # check exact match or if path is under an omitted directory
    return any(path == o or path.startswith(o.rstrip('/') + '/') for o in omit_paths)

# resolve repo path and omit file path
repo = CONFIG["repo_path"]
omit_path = Path(repo) / CONFIG["omit_file"]

# read omit file, strip whitespace, remove empty lines and comment lines, strip leading slashes
omit_paths = {p.strip().lstrip('/') for p in omit_path.read_text().splitlines() if p.strip() and not p.lstrip().startswith('#')}

# get one-line log of all commits on source branch
log = run(f"git log --oneline {CONFIG['source_branch']}", cwd=repo)

# parse each log line into (hash, message) pairs
commits = [line.split(" ", 1) for line in log.splitlines()]

clean = []
for hash_, msg in commits:
    # get list of files and their change status modified by this commit
    files_raw = run(f"git diff-tree --no-commit-id -r --name-status {hash_}", cwd=repo)
    # parse into {path: status} dict
    touched = {}
    for line in files_raw.splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            touched[parts[1].strip()] = parts[0].strip()
    # if no touched file overlaps with omit list, commit is safe to cherry-pick
    if not any(is_omitted(p, omit_paths) for p in touched.keys()):
        clean.append((hash_, msg, touched))

# print all cherry-pickable commits with touched files and status
print(f"Cherry-pickable commits ({len(clean)}):")
print(f"{'HASH':<10}\t{'MESSAGE':<50}\t{'STATUS'}\t{'FILE'}")
for hash_, msg, touched in clean:
    items = sorted(touched.items())
    if not items:
        print(f"{hash_:<10}\t{msg:<50}\t{'':6}\t''")
        continue
    print(f"{hash_:<10}\t{msg:<50}\t{items[0][1]:<6}\t{items[0][0]}")
    for path, status in items[1:]:
        print(f"{'':10}\t{'':50}\t{status:<6}\t{path}")

if not CONFIG["dry_run"]:
    # switch to target branch
    run(f"git switch {CONFIG['target_branch']}", cwd=repo)
    # cherry-pick in chronological order (reversed log order)
    for hash_, msg, _ in reversed(clean):
        print(f"Cherry-picking {hash_} {msg}")
        run(f"git cherry-pick {hash_}", cwd=repo)