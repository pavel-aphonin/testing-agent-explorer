#!/usr/bin/env bash
# PER-175: stop all 13 module bus-runners started by start-bus-runners.sh.
set -uo pipefail

ROLES=(
  SCREEN_PARSER DYNAMIC_PERCEIVER CONTEXT_IDENTIFIER AMBIGUITY_RESOLVER
  PLANNER SAFETY_GUARD REFLECTION REWARD_CRITIC PLATFORM_ADAPTER
  SCREEN_SEEKER GROUNDER GROUNDING_VERIFIER MEMORY
)

stopped=0
for role in "${ROLES[@]}"; do
  pidfile="/tmp/ta-bus-${role}.pid"
  if [[ -f "$pidfile" ]]; then
    pid="$(<"$pidfile")"
    if kill "$pid" 2>/dev/null; then
      echo "  stopped $role (pid $pid)"
      stopped=$((stopped+1))
    fi
    rm -f "$pidfile"
  fi
done
# Belt-and-braces: kill any stragglers by module path.
pkill -f "explorer.bus.runner" 2>/dev/null || true
echo "=== stopped $stopped runners ==="
