import importlib


def test_config_defaults_to_safe_local_runtime(monkeypatch):
    monkeypatch.delenv("FLASK_DEBUG", raising=False)
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    monkeypatch.delenv("EXPOSE_INTERNAL_ERRORS", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    monkeypatch.delenv("MARKET_DATA_PROVIDER", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)

    import app.config as config_module

    config_module = importlib.reload(config_module)

    assert config_module.Config.DEBUG is False
    assert config_module.Config.SECRET_KEY is None
    assert config_module.Config.CORS_ORIGINS == [
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ]
    assert config_module.Config.EXPOSE_INTERNAL_ERRORS is False
    assert config_module.Config.NEO4J_PASSWORD is None
    assert config_module.Config.MARKET_DATA_PROVIDER == "none"


def test_default_bind_host_is_loopback(monkeypatch):
    monkeypatch.delenv("FLASK_HOST", raising=False)

    import run

    assert run.get_bind_host() == "127.0.0.1"
