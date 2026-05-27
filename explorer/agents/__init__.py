"""PER-196: per-role agent wrappers.

Each agent is a thin facade over the OpenAI-compatible llama-server
identified by ``RoleResolver.resolve(role)``. The agent owns the
role's system prompt + response post-processing; the resolver owns
the endpoint URL / model name discovery.

Roles that don't have a GGUF model in the current roster (SCREEN_PARSER,
DYNAMIC_PERCEIVER, CONTEXT_IDENTIFIER — those need PyTorch wrappers
that we're not building this iteration) are NOT exposed here. The
hot-path code falls through to the existing monolith logic when the
agent for an unassigned role is asked for.
"""

from explorer.agents.ambiguity import AmbiguityAgent
from explorer.agents.base import RoleAgent, RoleAgentResult
from explorer.agents.grounder import GrounderAgent
from explorer.agents.memory import MemoryAgent
from explorer.agents.planner import PlannerAgent
from explorer.agents.safety import SafetyAgent

__all__ = [
    "AmbiguityAgent",
    "GrounderAgent",
    "MemoryAgent",
    "PlannerAgent",
    "RoleAgent",
    "RoleAgentResult",
    "SafetyAgent",
]
