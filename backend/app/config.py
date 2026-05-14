"""
Configuration Management
Loads configuration from .env file in project root directory
"""

import os
from dotenv import load_dotenv

# Load .env file from project root
# Path: MiroFish/.env (relative to backend/app/config.py)
project_root_env = os.path.join(os.path.dirname(__file__), '../../.env')

if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    # If no .env in root, try to load environment variables (for production)
    load_dotenv(override=True)


class Config:
    """Flask configuration class"""

    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mirofish-secret-key')
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'

    # JSON configuration - disable ASCII escaping to display Chinese directly (not as \uXXXX)
    JSON_AS_ASCII = False

    # LLM configuration (unified OpenAI format)
    LLM_DEFAULT_PROVIDER = os.environ.get('LLM_DEFAULT_PROVIDER', 'ollama')
    LLM_API_KEY = os.environ.get('LLM_API_KEY')
    LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'http://localhost:11434/v1')
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'qwen2.5:32b')

    # Neo4j configuration
    NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', 'mirofish')

    # Embedding configuration
    EMBEDDING_PROVIDER = os.environ.get('EMBEDDING_PROVIDER', 'ollama').strip().lower()
    EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'nomic-embed-text')
    EMBEDDING_BASE_URL = os.environ.get('EMBEDDING_BASE_URL', 'http://localhost:11434')
    EMBEDDING_DIMENSIONS = int(os.environ.get('EMBEDDING_DIMENSIONS', '768'))
    EMBEDDING_AUTO_PULL = os.environ.get('EMBEDDING_AUTO_PULL', 'True').lower() == 'true'
    EMBEDDING_CIRCUIT_BREAKER_SECONDS = int(os.environ.get('EMBEDDING_CIRCUIT_BREAKER_SECONDS', '60'))
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    GEMINI_EMBEDDING_MODEL = os.environ.get('GEMINI_EMBEDDING_MODEL', 'gemini-embedding-2')
    GEMINI_EMBEDDING_BASE_URL = os.environ.get(
        'GEMINI_EMBEDDING_BASE_URL',
        'https://generativelanguage.googleapis.com/v1beta'
    )
    STARTUP_STATUS_CHECK = os.environ.get('STARTUP_STATUS_CHECK', 'True').lower() == 'true'

    # File upload configuration
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt', 'markdown'}

    # Text processing configuration
    DEFAULT_CHUNK_SIZE = int(os.environ.get('GRAPH_BUILD_CHUNK_SIZE', '3000'))
    DEFAULT_CHUNK_OVERLAP = int(os.environ.get('GRAPH_BUILD_CHUNK_OVERLAP', '200'))
    GRAPH_BUILD_BATCH_SIZE = int(os.environ.get('GRAPH_BUILD_BATCH_SIZE', '8'))
    # 0 means auto: local Ollama stays conservative, cloud providers use more concurrency.
    GRAPH_BUILD_LLM_CONCURRENCY = int(os.environ.get('GRAPH_BUILD_LLM_CONCURRENCY', '0'))
    GRAPH_BUILD_LOCAL_LLM_CONCURRENCY = int(os.environ.get('GRAPH_BUILD_LOCAL_LLM_CONCURRENCY', '1'))
    GRAPH_BUILD_CLOUD_LLM_CONCURRENCY = int(os.environ.get('GRAPH_BUILD_CLOUD_LLM_CONCURRENCY', '3'))
    GRAPH_BUILD_BATCH_RETRIES = int(os.environ.get('GRAPH_BUILD_BATCH_RETRIES', '2'))
    GRAPH_BUILD_BATCH_RETRY_BASE_SECONDS = float(os.environ.get('GRAPH_BUILD_BATCH_RETRY_BASE_SECONDS', '2'))
    GRAPH_EXTRACTION_CACHE_ENABLED = os.environ.get('GRAPH_EXTRACTION_CACHE_ENABLED', 'True').lower() == 'true'
    GRAPH_EXTRACTION_CACHE_DIR = os.environ.get(
        'GRAPH_EXTRACTION_CACHE_DIR',
        os.path.join(UPLOAD_FOLDER, 'cache', 'ner_extractions')
    )

    # OASIS simulation configuration
    OASIS_DEFAULT_MAX_ROUNDS = int(os.environ.get('OASIS_DEFAULT_MAX_ROUNDS', '10'))
    OASIS_SIMULATION_DATA_DIR = os.path.join(os.path.dirname(__file__), '../uploads/simulations')

    # OASIS platform available actions configuration
    OASIS_TWITTER_ACTIONS = [
        'CREATE_POST', 'LIKE_POST', 'REPOST', 'FOLLOW', 'DO_NOTHING', 'QUOTE_POST'
    ]
    OASIS_REDDIT_ACTIONS = [
        'LIKE_POST', 'DISLIKE_POST', 'CREATE_POST', 'CREATE_COMMENT',
        'LIKE_COMMENT', 'DISLIKE_COMMENT', 'SEARCH_POSTS', 'SEARCH_USER',
        'TREND', 'REFRESH', 'DO_NOTHING', 'FOLLOW', 'MUTE'
    ]

    # Report Agent configuration
    REPORT_AGENT_MAX_TOOL_CALLS = int(os.environ.get('REPORT_AGENT_MAX_TOOL_CALLS', '5'))
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(os.environ.get('REPORT_AGENT_MAX_REFLECTION_ROUNDS', '2'))
    REPORT_AGENT_TEMPERATURE = float(os.environ.get('REPORT_AGENT_TEMPERATURE', '0.5'))
    REPORT_AGENT_INTERVIEW_TIMEOUT = float(os.environ.get('REPORT_AGENT_INTERVIEW_TIMEOUT', '240'))

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        errors = []
        if not cls.LLM_API_KEY:
            errors.append("LLM_API_KEY not configured (set to any non-empty value, e.g. 'ollama')")
        if not cls.NEO4J_URI:
            errors.append("NEO4J_URI not configured")
        if not cls.NEO4J_PASSWORD:
            errors.append("NEO4J_PASSWORD not configured")
        return errors
