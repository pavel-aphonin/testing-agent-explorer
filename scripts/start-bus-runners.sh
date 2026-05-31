#!/usr/bin/env bash
# PER-175 full integration: start ALL 13 module bus-runners.
#
# Each runner is one process: `python -m explorer.bus.runner --role <ROLE>`.
# It connects to Redis (TA_REDIS_URL), ensures its consumer group on its
# input stream, and loops consume→handle→publish. Code-only roles
# (PLATFORM_ADAPTER, SCREEN_SEEKER) run pure logic — no model endpoint.
#
# The host model servers (llama-server for Planner/Grounder, the perception
# service for Screen Parser / Dynamic Perceiver / Context Identifier, etc.)
# are started separately by testing-agent-infra/scripts/start-host-services.sh
# and the per-module llama-servers. This script only starts the bus glue.
#
# Usage:
#   TA_WORKER_TOKEN=... ./scripts/start-bus-runners.sh
#   ./scripts/start-bus-runners.sh            # reads token from infra/.env
#
# Logs: /tmp/ta-bus-<ROLE>.log   PIDs: /tmp/ta-bus-<ROLE>.pid
# Stop: ./scripts/stop-bus-runners.sh

set -euo pipefail
cd "$(dirname "$0")/.."

VENV_PY="$(pwd)/.venv/bin/python"
BACKEND_URL="${TA_BACKEND_URL:-http://localhost:8000}"
INFRA_ENV="$(cd ../testing-agent-infra 2>/dev/null && pwd)/.env"

# WORKER_TOKEN: env first, else infra/.env
if [[ -z "${TA_WORKER_TOKEN:-}" && -f "$INFRA_ENV" ]]; then
  TA_WORKER_TOKEN="$(grep -E '^WORKER_TOKEN=' "$INFRA_ENV" | cut -d= -f2- || true)"
fi
if [[ -z "${TA_WORKER_TOKEN:-}" ]]; then
  echo "ERROR: TA_WORKER_TOKEN not set and not found in $INFRA_ENV" >&2
  exit 1
fi

# All 13 modules. The two code-only roles (PLATFORM_ADAPTER, SCREEN_SEEKER)
# are included — their runners drive pure-code handlers.
ROLES=(
  SCREEN_PARSER
  DYNAMIC_PERCEIVER
  CONTEXT_IDENTIFIER
  AMBIGUITY_RESOLVER
  PLANNER
  SAFETY_GUARD
  REFLECTION
  REWARD_CRITIC
  PLATFORM_ADAPTER
  SCREEN_SEEKER
  GROUNDER
  GROUNDING_VERIFIER
  MEMORY
)

echo "=== Starting ${#ROLES[@]} bus runners (backend=$BACKEND_URL) ==="
for role in "${ROLES[@]}"; do
  pidfile="/tmp/ta-bus-${role}.pid"
  logfile="/tmp/ta-bus-${role}.log"
  if [[ -f "$pidfile" ]] && kill -0 "$(<"$pidfile")" 2>/dev/null; then
    echo "  ✓ $role already running (pid $(<"$pidfile"))"
    continue
  fi
  TA_BACKEND_URL="$BACKEND_URL" TA_WORKER_TOKEN="$TA_WORKER_TOKEN" \
    "$VENV_PY" -m explorer.bus.runner \
      --role "$role" \
      --backend-url "$BACKEND_URL" \
      --worker-token "$TA_WORKER_TOKEN" \
      > "$logfile" 2>&1 &
  echo "$!" > "$pidfile"
  sleep 0.3
  if kill -0 "$!" 2>/dev/null; then
    echo "  ✓ $role started (pid $!, log $logfile)"
  else
    echo "  ✗ $role failed — see $logfile"
  fi
done

echo ""
echo "=== Done. 13 runners launched. ==="
echo "Tail one:  tail -f /tmp/ta-bus-CONTEXT_IDENTIFIER.log"
echo "Stop all:  ./scripts/stop-bus-runners.sh"
