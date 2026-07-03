"""Market-data API: resolve a symbol's current value + suggested numeric bands.

Used by the macro target builder in the UI to auto-fill the anchor and outcome
bands. Degrades gracefully (success=false) so the UI can fall back to manual entry
when market data is disabled or unavailable.
"""

from flask import request, jsonify

from . import market_bp
from ..utils import market_data
from ..utils.logger import get_logger

logger = get_logger("mirofish.api.market")


@market_bp.route('/resolve', methods=['GET'])
def resolve():
    symbol = (request.args.get('symbol') or '').strip()
    kind = (request.args.get('kind') or 'price').strip().lower()
    horizon = (request.args.get('horizon') or '1m').strip()

    if not symbol:
        return jsonify({"success": False, "error": "Please provide a symbol"}), 400
    if kind not in {"price", "indicator"}:
        return jsonify({"success": False, "error": "kind must be 'price' or 'indicator'"}), 400
    if not market_data.is_enabled():
        return jsonify({
            "success": False,
            "error": "Market data is disabled (MARKET_DATA_PROVIDER=none). Enter values manually.",
            "disabled": True,
        }), 200

    try:
        data = market_data.resolve_target(symbol, kind=kind, horizon=horizon)
        return jsonify({"success": True, "data": data})
    except Exception as exc:
        logger.warning("market resolve failed for %s (%s): %s", symbol, kind, exc)
        return jsonify({"success": False, "error": str(exc)}), 200


@market_bp.route('/status', methods=['GET'])
def status():
    from ..config import Config
    return jsonify({"success": True, "data": {
        "enabled": market_data.is_enabled(),
        "provider": Config.MARKET_DATA_PROVIDER,
        "fred_configured": bool(Config.FRED_API_KEY),
    }})
