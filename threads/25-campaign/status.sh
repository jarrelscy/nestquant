#!/bin/bash
# T25 campaign status (read-only): ./status.sh [--alerts N] [--all]
exec /home/coder/git/glm52/.venv/bin/python "$(dirname "$(readlink -f "$0")")/nq25_status.py" "$@"
