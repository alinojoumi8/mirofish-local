"""
API Routes Module
"""

from flask import Blueprint

graph_bp = Blueprint('graph', __name__)
simulation_bp = Blueprint('simulation', __name__)
report_bp = Blueprint('report', __name__)
status_bp = Blueprint('status', __name__)
market_bp = Blueprint('market', __name__)

from . import graph  # noqa: E402, F401
from . import simulation  # noqa: E402, F401
from . import report  # noqa: E402, F401
from . import status  # noqa: E402, F401
from . import market  # noqa: E402, F401

