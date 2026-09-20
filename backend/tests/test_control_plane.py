import logging

import pytest
from flask import Flask

from app.services.control_plane import initialize_control_plane


def test_configured_database_failure_stops_startup(tmp_path):
    app = Flask(__name__)
    # A directory cannot be opened as a SQLite database.
    app.config['CONTROL_DB_PATH'] = str(tmp_path)
    app.config['OASIS_SIMULATION_DATA_DIR'] = str(tmp_path / 'simulations')
    with pytest.raises(RuntimeError, match='Control database initialization failed'):
        initialize_control_plane(app, logging.getLogger('test'), log_startup=False)
