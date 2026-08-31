# Shared location and interpreter discovery for the scripts in this directory.
#
# Source it, do not run it:
#
#   source "$(dirname "${BASH_SOURCE[0]}")/_repo.sh"
#
# Sets ROOT to the repository and PY to the interpreter to run with, and leaves
# either alone if the caller exported it. Both used to be absolute paths written
# into each script, which named the right things on exactly one computer and
# nothing at all anywhere else.
#
# The search order matches start-backend.sh: an explicit override, the active
# virtualenv, the one setup-engine.sh builds inside the repository, then a venv
# beside it.

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

if [[ -z ${PY:-} ]]; then
  # Built conditionally rather than as one list: an unset VIRTUAL_ENV expands to
  # nothing, and "/bin/python" exists on plenty of machines, so a list would
  # quietly select the system interpreter over the venv beside the repository.
  candidates=()
  if [[ -n ${VIRTUAL_ENV:-} ]]; then
    candidates+=("$VIRTUAL_ENV/bin/python")
  fi
  candidates+=("$ROOT/.venv/bin/python" "$ROOT/../venv/bin/python")
  for candidate in "${candidates[@]}"; do
    if [[ -x $candidate ]]; then
      PY=$candidate
      break
    fi
  done
fi
PY=${PY:-python3}
