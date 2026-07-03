"""Derive an agent's stance toward the prediction target from graph relationships.

For a realistic simulation each agent must embody its real-world position
(supportive / opposing / mixed / neutral) instead of a generic "neutral"
forecaster. That position is already present in the knowledge graph as
SUPPORTS / OPPOSES-style edges extracted from the source document, so we read it
directly rather than guessing. The result feeds three places:
  - persona text + LLM persona prompt (oasis_profile_generator)
  - the agent_config `stance` / `sentiment_bias` (simulation_config_generator)
  - the simulation-grounded forecast aggregation (forecasting)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Relationship names (edge_name) that signal a direction of support or opposition.
_SUPPORT_RELATIONS = {
    "SUPPORTS", "SUPPORT", "ENDORSES", "ENDORSE", "BACKS", "BACK", "FAVORS", "FAVOR",
    "APPROVES", "APPROVE", "CHAMPIONS", "CHAMPION", "PROMOTES", "PROMOTE", "ADVOCATES",
    "ADVOCATE", "PRAISES", "PRAISE", "DEFENDS", "DEFEND", "FUNDS", "SPONSORS",
    "ALLIES_WITH", "VOTES_FOR", "APPROVED",
}
_OPPOSE_RELATIONS = {
    "OPPOSES", "OPPOSE", "REJECTS", "REJECT", "CRITICIZES", "CRITICIZE", "WARNS", "WARN",
    "BLOCKS", "BLOCK", "RESISTS", "RESIST", "CONDEMNS", "CONDEMN", "FIGHTS", "FIGHT",
    "CHALLENGES", "CHALLENGE", "DISPUTES", "DISPUTE", "SUES", "PROTESTS", "PROTEST",
    "VETOES", "VETO", "VOTES_AGAINST",
}

# Lexical cues for free-text facts when the relation name itself is generic
# (e.g. VOTES_ON, ANNOUNCES, MONITORS).
_SUPPORT_WORDS = (
    "support", "endorse", "backed", "in favor", "favour", "favor", "approve", "champion",
    "praise", "advocate", "benefit", "welcome", "applaud", "celebrate", "voted in favor",
    "voted for", "yes vote",
)
_OPPOSE_WORDS = (
    "oppose", "reject", "critic", "warn", "against", "block", "resist", "condemn", "hurt",
    "harm", "concern", "burden", "fight", "protest", "overturn", "repeal", "dissent",
    "voted against", "no vote", "unfair",
)

_ROLE_ATTR_KEYS = ("title", "role", "position", "occupation", "job", "office")

_STANCE_VERB = {
    "supportive": "supports",
    "opposing": "opposes",
    "mixed": "has a mixed position on",
    "neutral": "is neutral on",
}


def entity_role(entity: Any) -> str:
    """Best-effort human role/title for an entity, falling back to its type."""
    attributes = getattr(entity, "attributes", None) or {}
    for key in _ROLE_ATTR_KEYS:
        value = attributes.get(key)
        if value and str(value).strip():
            return str(value).strip()
    etype = entity.get_entity_type() if hasattr(entity, "get_entity_type") else None
    return etype or "participant"


def stance_verb(stance: str) -> str:
    return _STANCE_VERB.get(stance, "is engaged with")


def _keywords(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) > 3}


def infer_stance(entity: Any, prediction_target: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Infer an agent's stance toward the prediction target from its graph edges.

    Returns ``{stance, sentiment_bias, rationale, support_score, oppose_score}``
    where stance ∈ {supportive, opposing, mixed, neutral} and
    sentiment_bias ∈ [-1.0, 1.0].
    """
    edges: List[Dict[str, Any]] = list(getattr(entity, "related_edges", None) or [])
    target_kw = _keywords((prediction_target or {}).get("question", "")) if prediction_target else set()

    support_score = 0.0
    oppose_score = 0.0
    best_support = ("", 0.0)
    best_oppose = ("", 0.0)

    for edge in edges:
        name = str(edge.get("edge_name", "")).upper().strip()
        fact = str(edge.get("fact", "")).strip()
        fact_l = fact.lower()

        # Edges whose fact overlaps the prediction target are weighted higher.
        relevance = 1.5 if (target_kw and (_keywords(fact) & target_kw)) else 1.0

        direction = 0
        weight = 0.0
        if name in _SUPPORT_RELATIONS:
            direction, weight = 1, 1.0
        elif name in _OPPOSE_RELATIONS:
            direction, weight = -1, 1.0
        else:
            has_sup = any(w in fact_l for w in _SUPPORT_WORDS)
            has_opp = any(w in fact_l for w in _OPPOSE_WORDS)
            if has_sup and not has_opp:
                direction, weight = 1, 0.5
            elif has_opp and not has_sup:
                direction, weight = -1, 0.5

        if direction == 0:
            continue

        score = weight * relevance
        if direction > 0:
            support_score += score
            if score > best_support[1]:
                best_support = (fact or name, score)
        else:
            oppose_score += score
            if score > best_oppose[1]:
                best_oppose = (fact or name, score)

    total = support_score + oppose_score
    if total <= 0:
        return {
            "stance": "neutral", "sentiment_bias": 0.0, "rationale": "",
            "support_score": 0.0, "oppose_score": 0.0,
        }

    net = support_score - oppose_score
    sentiment_bias = round(max(-1.0, min(1.0, net / total)), 2)

    if support_score > 0 and oppose_score > 0 and abs(net) / total < 0.34:
        stance = "mixed"
    elif net > 0:
        stance = "supportive"
    elif net < 0:
        stance = "opposing"
    else:
        stance = "mixed"

    if stance == "supportive":
        rationale = best_support[0]
    elif stance == "opposing":
        rationale = best_oppose[0]
    else:
        rationale = "; ".join(f for f in (best_support[0], best_oppose[0]) if f)

    return {
        "stance": stance,
        "sentiment_bias": sentiment_bias,
        "rationale": rationale[:300],
        "support_score": round(support_score, 2),
        "oppose_score": round(oppose_score, 2),
    }
