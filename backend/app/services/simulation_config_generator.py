"""
Simulation Configuration Intelligent Generator
Use LLM to automatically generate detailed simulation parameters based on simulation requirements, document content, and knowledge graph information
Implement full process automation without manual parameter setting

Adopt step-by-step generation strategy to avoid failures from generating too long content at once:
1. Generate time configuration
2. Generate event configuration
3. Generate agent configurations in batches
4. Generate platform configuration
"""

import json
import math
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime

from openai import OpenAI

from ..config import Config
from ..utils.logger import get_logger
from .entity_reader import EntityNode
from .forecasting import (
    DEFAULT_HORIZONS,
    DEFAULT_OUTCOMES,
    domain_guidance_for_forecast,
    normalize_forecast_settings,
)

logger = get_logger('mirofish.simulation_config')

# Time zone configuration for Chinese work schedules (Beijing Time)
CHINA_TIMEZONE_CONFIG = {
    # Dead hours (almost no activity)
    "dead_hours": [0, 1, 2, 3, 4, 5],
    # Morning hours (gradually waking up)
    "morning_hours": [6, 7, 8],
    # Work hours
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    # Evening peak (most active)
    "peak_hours": [19, 20, 21, 22],
    # Night hours (activity decreases)
    "night_hours": [23],
    # Activity multipliers
    "activity_multipliers": {
        "dead": 0.05,      # Almost no one in early morning
        "morning": 0.4,    # Gradually active in morning
        "work": 0.7,       # Medium activity during work hours
        "peak": 1.5,       # Evening peak
        "night": 0.5       # Activity decreases at night
    }
}


@dataclass
class AgentActivityConfig:
    """Activity configuration for a single Agent"""
    agent_id: int
    entity_uuid: str
    entity_name: str
    entity_type: str

    # Activity configuration (0.0-1.0)
    activity_level: float = 0.5  # Overall activity level

    # Speech frequency (expected posts per hour)
    posts_per_hour: float = 1.0
    comments_per_hour: float = 2.0

    # Active time periods (24-hour format, 0-23)
    active_hours: List[int] = field(default_factory=lambda: list(range(8, 23)))

    # Response speed (reaction delay to trending events, unit: simulation minutes)
    response_delay_min: int = 5
    response_delay_max: int = 60

    # Sentiment tendency (-1.0 to 1.0, negative to positive)
    sentiment_bias: float = 0.0

    # Stance (attitude toward specific topics)
    stance: str = "neutral"  # supportive, opposing, neutral, observer

    # Influence weight (determines probability of their speech being seen by other agents)
    influence_weight: float = 1.0


@dataclass
class TimeSimulationConfig:
    """Time simulation configuration (based on Chinese work schedule habits)"""
    # Total simulation time (simulation hours)
    total_simulation_hours: int = 72  # Default 72 hours (3 days)

    # Time represented per round (simulation minutes) - default 60 minutes (1 hour), speed up time
    minutes_per_round: int = 60

    # Range of agents activated per hour
    agents_per_hour_min: int = 5
    agents_per_hour_max: int = 20

    # Peak hours (evening 19-22, most active time for Chinese people)
    peak_hours: List[int] = field(default_factory=lambda: [19, 20, 21, 22])
    peak_activity_multiplier: float = 1.5

    # Off-peak hours (early morning 0-5, almost no activity)
    off_peak_hours: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    off_peak_activity_multiplier: float = 0.05  # Very low activity in early morning

    # Morning hours
    morning_hours: List[int] = field(default_factory=lambda: [6, 7, 8])
    morning_activity_multiplier: float = 0.4

    # Work hours
    work_hours: List[int] = field(default_factory=lambda: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    work_activity_multiplier: float = 0.7


@dataclass
class EventConfig:
    """Event configuration"""
    # Initial posts (triggering events at the start of simulation)
    initial_posts: List[Dict[str, Any]] = field(default_factory=list)

    # Scheduled events (events triggered at specific times)
    scheduled_events: List[Dict[str, Any]] = field(default_factory=list)

    # Hot topic keywords
    hot_topics: List[str] = field(default_factory=list)

    # Opinion narrative direction
    narrative_direction: str = ""

    # Scenario metadata
    scenario_pack: str = "baseline_adverse_favorable"


@dataclass
class PlatformConfig:
    """Platform-specific configuration"""
    platform: str  # twitter or reddit

    # Recommendation algorithm weights
    recency_weight: float = 0.4  # Time freshness
    popularity_weight: float = 0.3  # Popularity
    relevance_weight: float = 0.3  # Relevance

    # Viral threshold (number of interactions before triggering spread)
    viral_threshold: int = 10

    # Echo chamber effect strength (degree of similar opinion clustering)
    echo_chamber_strength: float = 0.5


@dataclass
class SimulationParameters:
    """Complete simulation parameter configuration"""
    # Basic information
    simulation_id: str
    project_id: str
    graph_id: str
    simulation_requirement: str

    # Time configuration
    time_config: TimeSimulationConfig = field(default_factory=TimeSimulationConfig)

    # Agent configuration list
    agent_configs: List[AgentActivityConfig] = field(default_factory=list)

    # Event configuration
    event_config: EventConfig = field(default_factory=EventConfig)

    # Platform configuration
    twitter_config: Optional[PlatformConfig] = None
    reddit_config: Optional[PlatformConfig] = None

    # Forecast configuration
    forecast_mode: str = "general"
    forecast_horizon: str = "medium_term"
    prediction_target: Dict[str, Any] = field(default_factory=dict)
    scenario_pack: str = "baseline_adverse_favorable"
    ensemble_runs: int = 5
    memory_mode: str = "practical"

    # LLM configuration
    llm_model: str = ""
    llm_base_url: str = ""

    # Generation metadata
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_reasoning: str = ""  # LLM reasoning explanation

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        time_dict = asdict(self.time_config)
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "time_config": time_dict,
            "agent_configs": [asdict(a) for a in self.agent_configs],
            "event_config": asdict(self.event_config),
            "twitter_config": asdict(self.twitter_config) if self.twitter_config else None,
            "reddit_config": asdict(self.reddit_config) if self.reddit_config else None,
            "forecast_mode": self.forecast_mode,
            "forecast_horizon": self.forecast_horizon,
            "prediction_target": self.prediction_target,
            "scenario_pack": self.scenario_pack,
            "ensemble_runs": self.ensemble_runs,
            "memory_mode": self.memory_mode,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "generated_at": self.generated_at,
            "generation_reasoning": self.generation_reasoning,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class SimulationConfigGenerator:
    """
    Simulation Configuration Intelligent Generator

    Use LLM to analyze simulation requirements, document content, knowledge graph entity information,
    and automatically generate optimal simulation parameter configuration

    Adopt step-by-step generation strategy:
    1. Generate time configuration and event configuration (lightweight)
    2. Generate agent configurations in batches (10-20 per batch)
    3. Generate platform configuration
    """

    # Maximum context length in characters
    MAX_CONTEXT_LENGTH = 50000
    # Number of agents per batch
    AGENTS_PER_BATCH = 15

    # Context truncation length for each step (characters)
    TIME_CONFIG_CONTEXT_LENGTH = 10000   # Time configuration
    EVENT_CONFIG_CONTEXT_LENGTH = 8000   # Event configuration
    ENTITY_SUMMARY_LENGTH = 300          # Entity summary
    AGENT_SUMMARY_LENGTH = 300           # Entity summary in agent configuration
    ENTITIES_PER_TYPE_DISPLAY = 20       # Number of entities to display per type

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model_name = model_name or Config.LLM_MODEL_NAME

        if not self.api_key:
            raise ValueError("LLM_API_KEY not configured")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def generate_config(
        self,
        simulation_id: str,
        project_id: str,
        graph_id: str,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode],
        enable_twitter: bool = True,
        enable_reddit: bool = True,
        forecast_settings: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> SimulationParameters:
        """
        Intelligently generate complete simulation configuration (step-by-step generation)

        Args:
            simulation_id: Simulation ID
            project_id: Project ID
            graph_id: Knowledge graph ID
            simulation_requirement: Simulation requirement description
            document_text: Original document content
            entities: Filtered entity list
            enable_twitter: Whether to enable Twitter
            enable_reddit: Whether to enable Reddit
            progress_callback: Progress callback function(current_step, total_steps, message)

        Returns:
            SimulationParameters: Complete simulation parameters
        """
        forecast = normalize_forecast_settings(
            forecast_settings or {},
            simulation_requirement=simulation_requirement,
            entity_types=[e.get_entity_type() or "Unknown" for e in entities],
        )
        forecast_dict = forecast.to_dict()
        logger.info(
            "Starting intelligent simulation configuration generation: "
            f"simulation_id={simulation_id}, entities={len(entities)}, forecast_mode={forecast.forecast_mode}"
        )
        
        # Calculate total steps
        num_batches = math.ceil(len(entities) / self.AGENTS_PER_BATCH)
        total_steps = 3 + num_batches  # time config + event config + N batch agents + platform config
        current_step = 0

        def report_progress(step: int, message: str):
            nonlocal current_step
            current_step = step
            if progress_callback:
                progress_callback(step, total_steps, message)
            logger.info(f"[{step}/{total_steps}] {message}")

        # 1. Build basic context information
        context = self._build_context(
            simulation_requirement=simulation_requirement,
            document_text=document_text,
            entities=entities,
            forecast_settings=forecast_dict,
        )
        
        reasoning_parts = []
        
        # ========== Step 1: Generate time configuration ==========
        report_progress(1, "Generating time configuration...")
        num_entities = len(entities)
        time_config_result = self._generate_time_config(context, num_entities, forecast_dict)
        time_config = self._parse_time_config(time_config_result, num_entities, forecast_dict)
        reasoning_parts.append(f"Time config: {time_config_result.get('reasoning', 'Success')}")

        # ========== Step 2: Generate event configuration ==========
        report_progress(2, "Generating event configuration and hot topics...")
        event_config_result = self._generate_event_config(context, simulation_requirement, entities, forecast_dict)
        event_config = self._parse_event_config(event_config_result, forecast_dict)
        reasoning_parts.append(f"Event config: {event_config_result.get('reasoning', 'Success')}")

        # ========== Step 3-N: Generate agent configurations in batches ==========
        all_agent_configs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.AGENTS_PER_BATCH
            end_idx = min(start_idx + self.AGENTS_PER_BATCH, len(entities))
            batch_entities = entities[start_idx:end_idx]

            report_progress(
                3 + batch_idx,
                f"Generating agent configuration ({start_idx + 1}-{end_idx}/{len(entities)})..."
            )
            
            batch_configs = self._generate_agent_configs_batch(
                context=context,
                entities=batch_entities,
                start_idx=start_idx,
                simulation_requirement=simulation_requirement,
                forecast_settings=forecast_dict,
            )
            all_agent_configs.extend(batch_configs)
        
        reasoning_parts.append(f"Agent config: Successfully generated {len(all_agent_configs)}")

        # ========== Assign initial post agents ==========
        logger.info("Assigning appropriate publisher agents to initial posts...")
        event_config = self._assign_initial_post_agents(event_config, all_agent_configs)
        assigned_count = len([p for p in event_config.initial_posts if p.get("poster_agent_id") is not None])
        reasoning_parts.append(f"Initial posts assigned: {assigned_count} posts assigned publishers")

        # ========== Final step: Generate platform configuration ==========
        report_progress(total_steps, "Generating platform configuration...")
        twitter_config = None
        reddit_config = None
        
        if enable_twitter:
            twitter_config = PlatformConfig(
                platform="twitter",
                recency_weight=0.4,
                popularity_weight=0.3,
                relevance_weight=0.3,
                viral_threshold=10,
                echo_chamber_strength=0.5
            )
        
        if enable_reddit:
            reddit_config = PlatformConfig(
                platform="reddit",
                recency_weight=0.3,
                popularity_weight=0.4,
                relevance_weight=0.3,
                viral_threshold=15,
                echo_chamber_strength=0.6
            )
        
        # Build final parameters
        params = SimulationParameters(
            simulation_id=simulation_id,
            project_id=project_id,
            graph_id=graph_id,
            simulation_requirement=simulation_requirement,
            time_config=time_config,
            agent_configs=all_agent_configs,
            event_config=event_config,
            twitter_config=twitter_config,
            reddit_config=reddit_config,
            forecast_mode=forecast.forecast_mode,
            forecast_horizon=forecast.forecast_horizon,
            prediction_target=forecast.prediction_target,
            scenario_pack=forecast.scenario_pack,
            ensemble_runs=forecast.ensemble_runs,
            memory_mode=forecast.memory_mode,
            llm_model=self.model_name,
            llm_base_url=self.base_url,
            generation_reasoning=" | ".join(reasoning_parts)
        )
        
        logger.info(f"Simulation configuration generation complete: {len(params.agent_configs)} agent configurations")

        return params

    def _build_context(
        self,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode],
        forecast_settings: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build LLM context, truncate to maximum length"""

        # Entity summary
        entity_summary = self._summarize_entities(entities)

        # Build context
        context_parts = [
            f"## Simulation Requirements\n{simulation_requirement}",
            f"\n## Forecast Controls\n{domain_guidance_for_forecast(forecast_settings or {})}",
            f"\n## Entity Information ({len(entities)})\n{entity_summary}",
        ]

        current_length = sum(len(p) for p in context_parts)
        remaining_length = self.MAX_CONTEXT_LENGTH - current_length - 500  # Reserve 500 characters

        if remaining_length > 0 and document_text:
            doc_text = document_text[:remaining_length]
            if len(document_text) > remaining_length:
                doc_text += "\n...(document truncated)"
            context_parts.append(f"\n## Original Document Content\n{doc_text}")

        return "\n".join(context_parts)

    def _summarize_entities(self, entities: List[EntityNode]) -> str:
        """Generate entity summary"""
        lines = []

        # Group by type
        by_type: Dict[str, List[EntityNode]] = {}
        for e in entities:
            t = e.get_entity_type() or "Unknown"
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(e)

        for entity_type, type_entities in by_type.items():
            lines.append(f"\n### {entity_type} ({len(type_entities)})")
            # Use configured display quantity and summary length
            display_count = self.ENTITIES_PER_TYPE_DISPLAY
            summary_len = self.ENTITY_SUMMARY_LENGTH
            for e in type_entities[:display_count]:
                summary_preview = (e.summary[:summary_len] + "...") if len(e.summary) > summary_len else e.summary
                lines.append(f"- {e.name}: {summary_preview}")
            if len(type_entities) > display_count:
                lines.append(f"  ... and {len(type_entities) - display_count} more")

        return "\n".join(lines)
    
    def _call_llm_with_retry(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """LLM call with retry, including JSON repair logic"""
        import re

        max_attempts = 3
        last_error = None

        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7 - (attempt * 0.1)  # Lower temperature with each retry
                    # Don't set max_tokens, let LLM generate freely
                )

                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason

                # Check if output was truncated
                if finish_reason == 'length':
                    logger.warning(f"LLM output truncated (attempt {attempt+1})")
                    content = self._fix_truncated_json(content)

                # Try to parse JSON
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parsing failed (attempt {attempt+1}): {str(e)[:80]}")

                    # Try to fix JSON
                    fixed = self._try_fix_config_json(content)
                    if fixed:
                        return fixed

                    last_error = e

            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt+1}): {str(e)[:80]}")
                last_error = e
                import time
                time.sleep(2 * (attempt + 1))

        raise last_error or Exception("LLM call failed")
    
    def _fix_truncated_json(self, content: str) -> str:
        """Fix truncated JSON"""
        content = content.strip()

        # Count unclosed parentheses
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')

        # Check for unclosed strings
        if content and content[-1] not in '",}]':
            content += '"'

        # Close parentheses
        content += ']' * open_brackets
        content += '}' * open_braces

        return content

    def _try_fix_config_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Try to fix configuration JSON"""
        import re

        # Fix truncated case
        content = self._fix_truncated_json(content)

        # Extract JSON portion
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group()

            # Remove newlines in strings
            def fix_string(match):
                s = match.group(0)
                s = s.replace('\n', ' ').replace('\r', ' ')
                s = re.sub(r'\s+', ' ', s)
                return s

            json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string, json_str)

            try:
                return json.loads(json_str)
            except:
                # Try removing all control characters
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
                json_str = re.sub(r'\s+', ' ', json_str)
                try:
                    return json.loads(json_str)
                except:
                    pass

        return None
    
    def _generate_time_config(
        self,
        context: str,
        num_entities: int,
        forecast_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate time configuration"""
        # Use configured context truncation length
        context_truncated = context[:self.TIME_CONFIG_CONTEXT_LENGTH]

        # Calculate maximum allowed value (90% of agents)
        max_agents_allowed = max(1, int(num_entities * 0.9))

        mode = forecast_settings.get("forecast_mode", "general")
        prompt = f"""Based on the following simulation requirements, generate time simulation configuration.

{context_truncated}

## Task
Please generate time configuration JSON.

### Basic principles (for reference only, adjust flexibly based on event nature and participant characteristics):
- Forecast mode is {mode}; use the schedule that makes the forecast realistic, not a generic Chinese social-media schedule.
- Legal cases should emphasize court/business hours, filing deadlines, hearing windows, and settlement negotiation cycles.
- Market/economy cases should emphasize market hours, news release windows, policy announcements, earnings, and after-hours reaction.
- General public discourse can still use daytime/evening peaks when relevant.
- Example values below are references only; adjust based on participants and event nature.

### Return JSON format (no markdown)

Example:
{{
    "total_simulation_hours": 72,
    "minutes_per_round": 60,
    "agents_per_hour_min": 5,
    "agents_per_hour_max": 50,
    "peak_hours": [19, 20, 21, 22],
    "off_peak_hours": [0, 1, 2, 3, 4, 5],
    "morning_hours": [6, 7, 8],
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    "reasoning": "Explanation of time configuration for this event"
}}

Field description:
- total_simulation_hours (int): Total simulation time, 24-168 hours, short for breaking news, long for ongoing topics
- minutes_per_round (int): Time per round, 30-120 minutes, recommend 60 minutes
- agents_per_hour_min (int): Minimum agents activated per hour (range: 1-{max_agents_allowed})
- agents_per_hour_max (int): Maximum agents activated per hour (range: 1-{max_agents_allowed})
- peak_hours (int array): Peak hours, adjust based on event participants
- off_peak_hours (int array): Off-peak hours, usually late night/early morning
- morning_hours (int array): Morning hours
- work_hours (int array): Work hours
- reasoning (string): Brief explanation for this configuration"""

        system_prompt = "You are a probabilistic simulation expert. Return pure JSON format and choose domain-realistic timing."

        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"Time config LLM generation failed: {e}, using default configuration")
            return self._get_default_time_config(num_entities, forecast_settings)
    
    def _get_default_time_config(self, num_entities: int, forecast_settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get default time configuration for the forecast mode."""
        mode = (forecast_settings or {}).get("forecast_mode", "general")
        if mode in {"legal_case", "market_economy"}:
            return {
                "total_simulation_hours": 72,
                "minutes_per_round": 60,
                "agents_per_hour_min": max(1, num_entities // 20),
                "agents_per_hour_max": max(4, num_entities // 6),
                "peak_hours": [9, 10, 11, 14, 15, 16],
                "off_peak_hours": [0, 1, 2, 3, 4, 5, 22, 23],
                "morning_hours": [7, 8],
                "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17],
                "reasoning": f"Using default {mode} forecast schedule (court/market/business-hour weighted)"
            }
        return {
            "total_simulation_hours": 72,
            "minutes_per_round": 60,  # 1 hour per round, speed up time
            "agents_per_hour_min": max(1, num_entities // 15),
            "agents_per_hour_max": max(5, num_entities // 5),
            "peak_hours": [19, 20, 21, 22],
            "off_peak_hours": [0, 1, 2, 3, 4, 5],
            "morning_hours": [6, 7, 8],
            "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            "reasoning": "Using default Chinese work schedule configuration (1 hour per round)"
        }

    def _parse_time_config(
        self,
        result: Dict[str, Any],
        num_entities: int,
        forecast_settings: Optional[Dict[str, Any]] = None,
    ) -> TimeSimulationConfig:
        """Parse time configuration result and verify agents_per_hour doesn't exceed total agents"""
        defaults = self._get_default_time_config(num_entities, forecast_settings)
        # Get original values
        agents_per_hour_min = result.get("agents_per_hour_min", defaults["agents_per_hour_min"])
        agents_per_hour_max = result.get("agents_per_hour_max", defaults["agents_per_hour_max"])

        # Verify and correct: ensure not exceeding total agents
        if agents_per_hour_min > num_entities:
            logger.warning(f"agents_per_hour_min ({agents_per_hour_min}) exceeds total agents ({num_entities}), corrected")
            agents_per_hour_min = max(1, num_entities // 10)

        if agents_per_hour_max > num_entities:
            logger.warning(f"agents_per_hour_max ({agents_per_hour_max}) exceeds total agents ({num_entities}), corrected")
            agents_per_hour_max = max(agents_per_hour_min + 1, num_entities // 2)

        # Ensure min < max
        if agents_per_hour_min >= agents_per_hour_max:
            agents_per_hour_min = max(1, agents_per_hour_max // 2)
            logger.warning(f"agents_per_hour_min >= max, corrected to {agents_per_hour_min}")

        return TimeSimulationConfig(
            total_simulation_hours=result.get("total_simulation_hours", defaults["total_simulation_hours"]),
            minutes_per_round=result.get("minutes_per_round", defaults["minutes_per_round"]),
            agents_per_hour_min=agents_per_hour_min,
            agents_per_hour_max=agents_per_hour_max,
            peak_hours=result.get("peak_hours", defaults["peak_hours"]),
            off_peak_hours=result.get("off_peak_hours", defaults["off_peak_hours"]),
            off_peak_activity_multiplier=0.05,  # Almost no one in early morning
            morning_hours=result.get("morning_hours", defaults["morning_hours"]),
            morning_activity_multiplier=0.4,
            work_hours=result.get("work_hours", defaults["work_hours"]),
            work_activity_multiplier=0.7,
            peak_activity_multiplier=1.5
        )
    
    def _generate_event_config(
        self,
        context: str,
        simulation_requirement: str,
        entities: List[EntityNode],
        forecast_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate event configuration"""

        # Get available entity types list for LLM reference
        entity_types_available = list(set(
            e.get_entity_type() or "Unknown" for e in entities
        ))

        # List representative entity names for each type
        type_examples = {}
        for e in entities:
            etype = e.get_entity_type() or "Unknown"
            if etype not in type_examples:
                type_examples[etype] = []
            if len(type_examples[etype]) < 3:
                type_examples[etype].append(e.name)

        type_info = "\n".join([
            f"- {t}: {', '.join(examples)}"
            for t, examples in type_examples.items()
        ])

        # Use configured context truncation length
        context_truncated = context[:self.EVENT_CONFIG_CONTEXT_LENGTH]

        prompt = f"""Based on the following simulation requirements, generate event configuration.

Simulation Requirements: {simulation_requirement}

{context_truncated}

## Available Entity Types and Examples
{type_info}

## Task
Please generate event configuration JSON:
- Extract hot topic keywords
- Describe opinion development direction
- Design initial post content, **each post must specify poster_type (publisher type)**
- Design scheduled scenario shocks; each scheduled event must include round, hour, event_type, title, content, poster_type, and expected_effect.

**Important**: poster_type must be selected from the "Available Entity Types" above so initial posts can be assigned to appropriate agents for publishing.
Example: Official statements should be published by Official/University type, news by MediaOutlet, student opinions by Student type.

Forecast controls:
{domain_guidance_for_forecast(forecast_settings)}

Domain event guidance:
- legal_case: use filing deadlines, hearings, discovery disputes, expert reports, settlement offers, injunction decisions, and credibility surprises.
- market_economy: use CPI, rate decisions, earnings, liquidity shocks, regulatory news, demand shocks, supply-chain events, and market-media reactions.

Return JSON format (no markdown):
{{
    "hot_topics": ["keyword1", "keyword2", ...],
    "narrative_direction": "<description of opinion development direction>",
    "initial_posts": [
        {{"content": "post content", "poster_type": "entity type (must select from available types)"}},
        ...
    ],
    "scheduled_events": [
        {{"round": 3, "hour": 10, "event_type": "hearing|filing|rate_decision|earnings|news|shock", "title": "event title", "content": "public event content", "poster_type": "entity type", "expected_effect": "how this changes probabilities"}},
        ...
    ],
    "reasoning": "<brief explanation>"
}}"""

        system_prompt = "You are a scenario design expert for probabilistic forecasting. Return pure JSON format. poster_type must match available entity types precisely."

        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"Event config LLM generation failed: {e}, using default configuration")
            return self._get_default_event_config(forecast_settings)

    def _get_default_event_config(self, forecast_settings: Dict[str, Any]) -> Dict[str, Any]:
        mode = forecast_settings.get("forecast_mode", "general")
        target = forecast_settings.get("prediction_target", {})
        if mode == "legal_case":
            return {
                "hot_topics": ["evidence strength", "procedural leverage", "settlement posture"],
                "narrative_direction": "The simulation tests how evidence, procedure, and incentives change legal outcome probabilities.",
                "initial_posts": [],
                "scheduled_events": [
                    {
                        "round": 3,
                        "hour": 10,
                        "event_type": "procedural_update",
                        "title": "Procedural checkpoint",
                        "content": "A procedural checkpoint forces agents to reassess evidence strength, settlement leverage, and missing proof.",
                        "poster_type": "Court",
                        "expected_effect": "Updates probability ranges around merits and leverage.",
                    }
                ],
                "reasoning": "Using default legal forecast scenario events",
            }
        if mode == "market_economy":
            return {
                "hot_topics": ["demand signal", "liquidity", "policy shock"],
                "narrative_direction": "The simulation tests how demand, liquidity, and policy/news shocks change market probabilities.",
                "initial_posts": [],
                "scheduled_events": [
                    {
                        "round": 3,
                        "hour": 9,
                        "event_type": "market_data_release",
                        "title": "Market data checkpoint",
                        "content": "A fresh market data checkpoint forces agents to reassess base, upside, and downside cases.",
                        "poster_type": "Analyst",
                        "expected_effect": "Updates probability ranges around demand and liquidity.",
                    }
                ],
                "reasoning": "Using default market forecast scenario events",
            }
        return {
            "hot_topics": [],
            "narrative_direction": "The simulation tests alternative future paths from the uploaded evidence.",
            "initial_posts": [],
            "scheduled_events": [],
            "reasoning": "Using default forecast configuration",
        }

    def _parse_event_config(self, result: Dict[str, Any], forecast_settings: Dict[str, Any]) -> EventConfig:
        """Parse event configuration result"""
        return EventConfig(
            initial_posts=result.get("initial_posts", []),
            scheduled_events=result.get("scheduled_events", []),
            hot_topics=result.get("hot_topics", []),
            narrative_direction=result.get("narrative_direction", ""),
            scenario_pack=forecast_settings.get("scenario_pack", "baseline_adverse_favorable"),
        )
    
    def _assign_initial_post_agents(
        self,
        event_config: EventConfig,
        agent_configs: List[AgentActivityConfig]
    ) -> EventConfig:
        """
        Assign appropriate publisher agents to initial posts

        Match agent_id based on each post's poster_type
        """
        if not event_config.initial_posts and not event_config.scheduled_events:
            return event_config

        # Build agent index by entity type
        agents_by_type: Dict[str, List[AgentActivityConfig]] = {}
        for agent in agent_configs:
            etype = agent.entity_type.lower()
            if etype not in agents_by_type:
                agents_by_type[etype] = []
            agents_by_type[etype].append(agent)

        # Type mapping table (handle different formats LLM might output)
        type_aliases = {
            "official": ["official", "university", "governmentagency", "government"],
            "university": ["university", "official"],
            "mediaoutlet": ["mediaoutlet", "media"],
            "student": ["student", "person"],
            "professor": ["professor", "expert", "teacher"],
            "alumni": ["alumni", "person"],
            "organization": ["organization", "ngo", "company", "group"],
            "person": ["person", "student", "alumni"],
            "court": ["court", "judge", "governmentinstitution", "governmentagency", "organization"],
            "judge": ["judge", "court", "person"],
            "plaintiff": ["plaintiff", "claimant", "party", "person"],
            "defendant": ["defendant", "respondent", "party", "person"],
            "counsel": ["counsel", "lawyer", "attorney", "person"],
            "attorney": ["attorney", "lawyer", "counsel", "person"],
            "lawyer": ["lawyer", "attorney", "counsel", "person"],
            "analyst": ["analyst", "expert", "person", "mediaoutlet"],
            "investor": ["investor", "trader", "fund", "person", "organization"],
            "regulator": ["regulator", "governmentagency", "government", "organization"],
        }

        # Track used agent indices for each type to avoid reusing same agent
        used_indices: Dict[str, int] = {}

        def assign_posts(posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            updated_posts = []
            for post in posts:
                poster_type = post.get("poster_type", "").lower()
                content = post.get("content", "")

                # Try to find matching agent
                matched_agent_id = None

                # 1. Direct match
                if poster_type in agents_by_type:
                    agents = agents_by_type[poster_type]
                    idx = used_indices.get(poster_type, 0) % len(agents)
                    matched_agent_id = agents[idx].agent_id
                    used_indices[poster_type] = idx + 1
                else:
                    # 2. Match using aliases
                    for alias_key, aliases in type_aliases.items():
                        if poster_type in aliases or alias_key == poster_type:
                            for alias in aliases:
                                if alias in agents_by_type:
                                    agents = agents_by_type[alias]
                                    idx = used_indices.get(alias, 0) % len(agents)
                                    matched_agent_id = agents[idx].agent_id
                                    used_indices[alias] = idx + 1
                                    break
                        if matched_agent_id is not None:
                            break

                # 3. If still not found, use agent with highest influence
                if matched_agent_id is None:
                    logger.warning(f"No matching agent found for type '{poster_type}', using agent with highest influence")
                    if agent_configs:
                        # Sort by influence, select highest
                        sorted_agents = sorted(agent_configs, key=lambda a: a.influence_weight, reverse=True)
                        matched_agent_id = sorted_agents[0].agent_id
                    else:
                        matched_agent_id = 0

                updated = dict(post)
                updated["content"] = content
                updated["poster_type"] = post.get("poster_type", "Unknown")
                updated["poster_agent_id"] = matched_agent_id
                updated_posts.append(updated)

                logger.info(f"Post assigned: poster_type='{poster_type}' -> agent_id={matched_agent_id}")

            return updated_posts

        event_config.initial_posts = assign_posts(event_config.initial_posts)
        event_config.scheduled_events = assign_posts(event_config.scheduled_events)
        return event_config
    
    def _generate_agent_configs_batch(
        self,
        context: str,
        entities: List[EntityNode],
        start_idx: int,
        simulation_requirement: str,
        forecast_settings: Dict[str, Any],
    ) -> List[AgentActivityConfig]:
        """Generate agent configurations in batch"""

        # Build entity information (using configured summary length)
        entity_list = []
        summary_len = self.AGENT_SUMMARY_LENGTH
        for i, e in enumerate(entities):
            entity_list.append({
                "agent_id": start_idx + i,
                "entity_name": e.name,
                "entity_type": e.get_entity_type() or "Unknown",
                "summary": e.summary[:summary_len] if e.summary else ""
            })

        prompt = f"""Based on the following information, generate forecast simulation activity configuration for each entity.

Simulation Requirements: {simulation_requirement}

Forecast Controls:
{domain_guidance_for_forecast(forecast_settings)}

## Entity List
```json
{json.dumps(entity_list, ensure_ascii=False, indent=2)}
```

## Task
Generate activity configuration for each entity, noting:
- Use domain-realistic timing and roles, not generic public-chat behavior.
- Legal mode: courts/counsel/regulators are lower-frequency but high influence; parties and witnesses react around procedural events; settlement actors respond after delay.
- Market mode: policy actors and companies are lower-frequency/high-influence; analysts/media/traders react quickly; consumers/supply actors reveal demand signals.
- posts_per_hour and comments_per_hour should represent expected forecast-relevant actions, not generic chatter.
- stance must reflect role-specific forecast posture toward the prediction target.

Return JSON format (no markdown):
{{
    "agent_configs": [
        {{
            "agent_id": <must match input>,
            "activity_level": <0.0-1.0>,
            "posts_per_hour": <posting frequency>,
            "comments_per_hour": <comment frequency>,
            "active_hours": [<active hours list, consider Chinese work schedule>],
            "response_delay_min": <minimum response delay minutes>,
            "response_delay_max": <maximum response delay minutes>,
            "sentiment_bias": <-1.0 to 1.0>,
            "stance": "<supportive/opposing/neutral/observer>",
            "influence_weight": <influence weight>
        }},
        ...
    ]
}}"""

        system_prompt = "You are a forecast-agent behavior analysis expert. Return pure JSON with domain-realistic settings."

        try:
            result = self._call_llm_with_retry(prompt, system_prompt)
            llm_configs = {cfg["agent_id"]: cfg for cfg in result.get("agent_configs", [])}
        except Exception as e:
            logger.warning(f"Agent config batch LLM generation failed: {e}, using rule-based generation")
            llm_configs = {}

        # Build AgentActivityConfig objects
        configs = []
        for i, entity in enumerate(entities):
            agent_id = start_idx + i
            cfg = llm_configs.get(agent_id, {})

            # If LLM didn't generate, use rule-based generation
            if not cfg:
                cfg = self._generate_agent_config_by_rule(entity, forecast_settings)

            config = AgentActivityConfig(
                agent_id=agent_id,
                entity_uuid=entity.uuid,
                entity_name=entity.name,
                entity_type=entity.get_entity_type() or "Unknown",
                activity_level=cfg.get("activity_level", 0.5),
                posts_per_hour=cfg.get("posts_per_hour", 0.5),
                comments_per_hour=cfg.get("comments_per_hour", 1.0),
                active_hours=cfg.get("active_hours", list(range(9, 23))),
                response_delay_min=cfg.get("response_delay_min", 5),
                response_delay_max=cfg.get("response_delay_max", 60),
                sentiment_bias=cfg.get("sentiment_bias", 0.0),
                stance=cfg.get("stance", "neutral"),
                influence_weight=cfg.get("influence_weight", 1.0)
            )
            configs.append(config)

        return configs
    
    def _generate_agent_config_by_rule(self, entity: EntityNode, forecast_settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate single agent configuration based on rules."""
        entity_type = (entity.get_entity_type() or "Unknown").lower()
        mode = (forecast_settings or {}).get("forecast_mode", "general")

        if mode == "legal_case":
            if any(term in entity_type for term in ["court", "judge"]):
                return {"activity_level": 0.25, "posts_per_hour": 0.05, "comments_per_hour": 0.05, "active_hours": list(range(9, 17)), "response_delay_min": 120, "response_delay_max": 480, "sentiment_bias": 0.0, "stance": "neutral_decision_maker", "influence_weight": 3.0}
            if any(term in entity_type for term in ["plaintiff", "defendant", "claimant", "respondent", "party"]):
                return {"activity_level": 0.55, "posts_per_hour": 0.25, "comments_per_hour": 0.5, "active_hours": list(range(8, 21)), "response_delay_min": 30, "response_delay_max": 180, "sentiment_bias": 0.0, "stance": "self_interested", "influence_weight": 1.5}
            if any(term in entity_type for term in ["lawyer", "attorney", "counsel"]):
                return {"activity_level": 0.5, "posts_per_hour": 0.2, "comments_per_hour": 0.6, "active_hours": list(range(7, 20)), "response_delay_min": 15, "response_delay_max": 120, "sentiment_bias": 0.0, "stance": "strategic_advocate", "influence_weight": 2.2}
            if any(term in entity_type for term in ["expert", "witness"]):
                return {"activity_level": 0.35, "posts_per_hour": 0.15, "comments_per_hour": 0.35, "active_hours": list(range(9, 18)), "response_delay_min": 60, "response_delay_max": 240, "sentiment_bias": 0.0, "stance": "evidence_weighted", "influence_weight": 1.8}

        if mode == "market_economy":
            if any(term in entity_type for term in ["bank", "central", "regulator", "government"]):
                return {"activity_level": 0.25, "posts_per_hour": 0.1, "comments_per_hour": 0.05, "active_hours": list(range(8, 18)), "response_delay_min": 60, "response_delay_max": 240, "sentiment_bias": 0.0, "stance": "policy_guarded", "influence_weight": 3.0}
            if any(term in entity_type for term in ["analyst", "media", "journalist"]):
                return {"activity_level": 0.65, "posts_per_hour": 0.6, "comments_per_hour": 0.8, "active_hours": list(range(6, 23)), "response_delay_min": 5, "response_delay_max": 45, "sentiment_bias": 0.0, "stance": "signal_interpreter", "influence_weight": 2.2}
            if any(term in entity_type for term in ["investor", "trader", "fund"]):
                return {"activity_level": 0.8, "posts_per_hour": 0.5, "comments_per_hour": 1.2, "active_hours": list(range(7, 22)), "response_delay_min": 1, "response_delay_max": 20, "sentiment_bias": 0.0, "stance": "risk_adjusting", "influence_weight": 1.7}

        if entity_type in ["university", "governmentagency", "ngo"]:
            # Official institutions: work hour activity, low frequency, high influence
            return {
                "activity_level": 0.2,
                "posts_per_hour": 0.1,
                "comments_per_hour": 0.05,
                "active_hours": list(range(9, 18)),  # 9:00-17:59
                "response_delay_min": 60,
                "response_delay_max": 240,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 3.0
            }
        elif entity_type in ["mediaoutlet"]:
            # Media: all-day activity, medium frequency, high influence
            return {
                "activity_level": 0.5,
                "posts_per_hour": 0.8,
                "comments_per_hour": 0.3,
                "active_hours": list(range(7, 24)),  # 7:00-23:59
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "observer",
                "influence_weight": 2.5
            }
        elif entity_type in ["professor", "expert", "official"]:
            # Experts/Professors: work + evening activity, medium frequency
            return {
                "activity_level": 0.4,
                "posts_per_hour": 0.3,
                "comments_per_hour": 0.5,
                "active_hours": list(range(8, 22)),  # 8:00-21:59
                "response_delay_min": 15,
                "response_delay_max": 90,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 2.0
            }
        elif entity_type in ["student"]:
            # Students: mainly evening, high frequency
            return {
                "activity_level": 0.8,
                "posts_per_hour": 0.6,
                "comments_per_hour": 1.5,
                "active_hours": [8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23],  # Morning + evening
                "response_delay_min": 1,
                "response_delay_max": 15,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 0.8
            }
        elif entity_type in ["alumni"]:
            # Alumni: mainly evening
            return {
                "activity_level": 0.6,
                "posts_per_hour": 0.4,
                "comments_per_hour": 0.8,
                "active_hours": [12, 13, 19, 20, 21, 22, 23],  # Lunch break + evening
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0
            }
        else:
            # Ordinary people: evening peak
            return {
                "activity_level": 0.7,
                "posts_per_hour": 0.5,
                "comments_per_hour": 1.2,
                "active_hours": [9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23],  # Daytime + evening
                "response_delay_min": 2,
                "response_delay_max": 20,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0
            }
    

