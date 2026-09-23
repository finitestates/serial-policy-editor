#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/mutation_pipeline.sh [all|stage|results|browse]

Stages, in dependency order:
  sampling
  sampler_config
  episode_history
  episode_lineage
  episode_replay_source
  episode_lifecycle
  episode_engine

With no argument, run every stage in order. Mutmut keeps its cache in the
ignored .mutmut-workspace/ directory, so rerunning resumes prior work.

Set MUTMUT_MAX_CHILDREN to limit mutmut's parallel workers.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
workspace="$repo_root/.mutmut-workspace"
marker="$workspace/.created-by-mutation-pipeline"

if [[ -e "$workspace" && ! -f "$marker" ]]; then
    echo "Refusing to use $workspace because it is not marked as this pipeline's workspace." >&2
    exit 2
fi

if [[ ! -e "$workspace" ]]; then
    mkdir -p "$workspace"
    : > "$marker"
fi

# Mutmut needs a top-level src/ tree to generate the same dotted names that
# pytest records. Refresh a private copy so mutmut apply/browse cannot change
# core/src in the user's checkout.
rm -rf -- "$workspace/src"
cp -a "$repo_root/core/src" "$workspace/src"

link_workspace_path() {
    local name=$1
    local target=$2
    local path="$workspace/$name"

    if [[ -e "$path" && ! -L "$path" ]]; then
        echo "Refusing to replace non-symlink workspace path: $path" >&2
        exit 2
    fi
    ln -sfn "$target" "$path"
}

link_workspace_path tests ../tests
link_workspace_path core ../core
link_workspace_path setup.cfg ../mutation/setup.cfg
link_workspace_path mutmut_pytest.ini ../mutation/pytest.ini

if [[ -n "${MUTMUT_BIN:-}" ]]; then
    mutmut_bin=$(command -v "$MUTMUT_BIN" || true)
elif [[ -x "$repo_root/.venv/bin/mutmut" ]]; then
    mutmut_bin="$repo_root/.venv/bin/mutmut"
else
    mutmut_bin=$(command -v mutmut || true)
fi

if [[ -z "$mutmut_bin" ]]; then
    echo "mutmut was not found. Install it in .venv or put it on PATH." >&2
    exit 127
fi

if [[ "$mutmut_bin" != /* ]]; then
    mutmut_bin=$(cd -- "$(dirname -- "$mutmut_bin")" && pwd)/$(basename -- "$mutmut_bin")
fi

stage_pattern() {
    case "$1" in
        sampling) echo 'trajectory_editor.core.sampling.*' ;;
        sampler_config) echo 'trajectory_editor.core.sampler_config.*' ;;
        episode_history) echo 'trajectory_editor.episode_history.*' ;;
        episode_lineage) echo 'trajectory_editor.episode_lineage.*' ;;
        episode_replay_source) echo 'trajectory_editor.episode_replay_source.*' ;;
        episode_lifecycle) echo 'trajectory_editor.episode_lifecycle.*' ;;
        episode_engine) echo 'trajectory_editor.episode_engine.*' ;;
        *) return 1 ;;
    esac
}

run_stage() {
    local stage=$1
    local pattern
    pattern=$(stage_pattern "$stage") || {
        echo "Unknown mutation stage: $stage" >&2
        usage >&2
        exit 2
    }

    echo "=== Mutation stage: $stage ==="
    if [[ -n "${MUTMUT_MAX_CHILDREN:-}" ]]; then
        (cd -- "$workspace" && "$mutmut_bin" run --max-children "$MUTMUT_MAX_CHILDREN" "$pattern")
    else
        (cd -- "$workspace" && "$mutmut_bin" run "$pattern")
    fi
}

command_name=${1:-all}
case "$command_name" in
    all)
        for stage in sampling sampler_config episode_history episode_lineage \
            episode_replay_source episode_lifecycle episode_engine; do
            run_stage "$stage"
        done
        ;;
    results|browse)
        (cd -- "$workspace" && "$mutmut_bin" "$command_name")
        ;;
    *)
        if [[ $# -ne 1 ]]; then
            usage >&2
            exit 2
        fi
        run_stage "$command_name"
        ;;
esac
