#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
worktree_dir="$(mktemp -d "${TMPDIR:-/tmp}/job-agent-checkout.XXXXXX")"
cleanup() {
  git -C "$repo_root" worktree remove --force "$worktree_dir" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if [[ -n "$(git -C "$repo_root" status --porcelain)" ]]; then
  echo "Clean-checkout verification requires all intended files to be tracked and committed." >&2
  exit 1
fi

git -C "$repo_root" worktree add --detach "$worktree_dir" HEAD
cd "$worktree_dir"

required_files=(
  "README.md"
  "requirements.txt"
  "src/agent/graph.py"
  "src/tools/fit_analysis/tool.py"
  "src/tools/resume_tailoring/tool.py"
  "scripts/preflight.py"
  "scripts/run_production_smoke.py"
)
for path in "${required_files[@]}"; do
  test -f "$path" || { echo "Missing required tracked file: $path" >&2; exit 1; }
done

python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -c "import src.app.app; from src.tools.registry import load_tool_registry; assert len(load_tool_registry()) == 5"
.venv/bin/python -m pytest -q
