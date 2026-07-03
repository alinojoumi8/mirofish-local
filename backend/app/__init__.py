"""
MiroFish Backend - Flask Application Factory
"""

import os
import warnings
import time

# Suppress multiprocessing resource_tracker warnings (from third-party libraries like transformers)
# Must be set before all other imports
warnings.filterwarnings("ignore", message=".*resource_tracker.*")

from flask import Flask, g, request
from flask_cors import CORS

from .config import Config
from .utils.logger import setup_logger, get_logger


def create_app(config_class=Config):
    """Flask application factory function"""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Configure JSON encoding: ensure Chinese displays directly (not as \uXXXX)
    # Flask >= 2.3 uses app.json.ensure_ascii, older versions use JSON_AS_ASCII config
    if hasattr(app, 'json') and hasattr(app.json, 'ensure_ascii'):
        app.json.ensure_ascii = False

    # Setup logging
    logger = setup_logger('mirofish')

    # Only print startup info in reloader subprocess (avoid printing twice in debug mode)
    is_reloader_process = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
    debug_mode = app.config.get('DEBUG', False)
    should_log_startup = not debug_mode or is_reloader_process

    if should_log_startup:
        logger.info("=" * 50)
        logger.info("MiroFish-Offline Backend starting...")
        logger.info("=" * 50)

    # Enable CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # --- Initialize Neo4jStorage singleton (DI via app.extensions) ---
    from .storage import Neo4jStorage
    try:
        neo4j_storage = Neo4jStorage()
        app.extensions['neo4j_storage'] = neo4j_storage
        if should_log_startup:
            logger.info("Neo4jStorage initialized (connected to %s)", Config.NEO4J_URI)
            if Config.STARTUP_STATUS_CHECK:
                status = neo4j_storage.health_status()
                embedding = status.get("embedding", {})
                logger.info(
                    "Startup status: neo4j=%s embedding=%s vector_search_usable=%s",
                    status.get("healthy"),
                    embedding.get("healthy"),
                    status.get("vector_search_usable"),
                )
                if not embedding.get("healthy"):
                    logger.warning("Embedding provider is not ready: %s", embedding.get("error"))
    except Exception as e:
        logger.error("Neo4jStorage initialization failed: %s", e)
        # Store None so endpoints can return 503 gracefully
        app.extensions['neo4j_storage'] = None

    # Register simulation process cleanup function (ensure all simulation processes terminate on server shutdown)
    from .services.simulation_runner import SimulationRunner
    SimulationRunner.register_cleanup()
    if should_log_startup:
        logger.info("Simulation process cleanup function registered")

    # Request logging middleware
    @app.before_request
    def log_request():
        g.request_started_at = time.perf_counter()
        logger = get_logger('mirofish.request')
        logger.debug("Request started: %s %s", request.method, request.path)
        if request.content_type and 'json' in request.content_type:
            payload = request.get_json(silent=True)
            if isinstance(payload, dict):
                logger.debug(
                    "Request JSON keys: %s %s keys=%s",
                    request.method,
                    request.path,
                    ",".join(sorted(payload.keys())),
                )
            else:
                logger.debug("Request JSON body: %s %s body=%s", request.method, request.path, payload)

    @app.after_request
    def log_response(response):
        logger = get_logger('mirofish.request')
        started_at = getattr(g, "request_started_at", None)
        duration_ms = (time.perf_counter() - started_at) * 1000 if started_at else 0
        message = "HTTP %s %s -> %s duration_ms=%.1f"
        args = (request.method, request.path, response.status_code, duration_ms)
        if response.status_code >= 500:
            logger.error(message, *args)
        elif response.status_code >= 400:
            logger.warning(message, *args)
        else:
            logger.debug(message, *args)
        return response

    @app.teardown_request
    def log_unhandled_exception(exc):
        if exc is not None:
            logger = get_logger('mirofish.request')
            logger.error(
                "Unhandled exception during %s %s",
                request.method,
                request.path,
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    # Register blueprints
    from .api import graph_bp, simulation_bp, report_bp, status_bp, market_bp
    app.register_blueprint(status_bp, url_prefix='/api')
    app.register_blueprint(graph_bp, url_prefix='/api/graph')
    app.register_blueprint(simulation_bp, url_prefix='/api/simulation')
    app.register_blueprint(report_bp, url_prefix='/api/report')
    app.register_blueprint(market_bp, url_prefix='/api/market')

    # Health check
    @app.route('/health')
    def health():
        return {'status': 'ok', 'service': 'MiroFish-Offline Backend'}

    if should_log_startup:
        logger.info("MiroFish-Offline Backend startup complete")

    return app
