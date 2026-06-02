from routes.voice import voice_bp
from routes.risk import risk_bp
from routes.sos import sos_bp


def register_routes(app):
    app.register_blueprint(voice_bp)
    app.register_blueprint(risk_bp)
    app.register_blueprint(sos_bp)
