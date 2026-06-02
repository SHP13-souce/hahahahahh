from datetime import datetime, timezone
from extensions import db


class SOSEvent(db.Model):
    __tablename__ = "sos_events"

    id = db.Column(db.Integer, primary_key=True)
    sos_id = db.Column(db.String(30), unique=True, nullable=False, index=True)

    # Requester
    user_id = db.Column(db.String(50), nullable=False)
    user_name = db.Column(db.String(100), nullable=False)

    # Location
    location_lng = db.Column(db.Float, nullable=False)
    location_lat = db.Column(db.Float, nullable=False)
    location_address = db.Column(db.String(300), nullable=True)

    # Risk assessment
    risk_score = db.Column(db.Integer, nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)
    risk_type = db.Column(db.String(50), nullable=False)
    risk_reasons = db.Column(db.JSON, nullable=True)
    risk_suggestion = db.Column(db.String(300), nullable=True)

    # Trigger source
    source = db.Column(db.String(50), nullable=False)

    # Status: pending | responding | arrived | cancelled | closed
    status = db.Column(db.String(20), nullable=False, default="pending")

    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationship: one SOS can have many responders
    responders = db.relationship(
        "Responder", back_populates="sos_event", lazy=True,
        order_by="Responder.responded_at"
    )

    def to_dict(self):
        result = {
            "sosId": self.sos_id,
            "userId": self.user_id,
            "userName": self.user_name,
            "location": {
                "lng": self.location_lng,
                "lat": self.location_lat,
                "address": self.location_address or "",
            },
            "risk": {
                "riskScore": self.risk_score,
                "riskLevel": self.risk_level,
                "riskType": self.risk_type,
                "reasons": [r["label"] for r in (self.risk_reasons or [])],
                "suggestion": self.risk_suggestion or "",
            },
            "status": self.status,
            "createdAt": (
                self.created_at.strftime("%Y-%m-%d %H:%M:%S")
                if self.created_at
                else ""
            ),
            "responders": [
                {
                    "volunteerId": r.volunteer_id,
                    "volunteerName": r.volunteer_name,
                    "volunteerType": r.volunteer_type,
                    "phone": r.phone,
                    "etaMinutes": r.eta_minutes,
                    "status": r.status,
                }
                for r in (self.responders or [])
            ],
        }
        return result


class Responder(db.Model):
    __tablename__ = "responders"

    id = db.Column(db.Integer, primary_key=True)
    sos_id = db.Column(
        db.String(30), db.ForeignKey("sos_events.sos_id"), nullable=False, index=True
    )

    volunteer_id = db.Column(db.String(50), nullable=False)
    volunteer_name = db.Column(db.String(100), nullable=False)
    volunteer_type = db.Column(db.String(50), nullable=True)
    phone = db.Column(db.String(30), nullable=True)
    location_lng = db.Column(db.Float, nullable=True)
    location_lat = db.Column(db.Float, nullable=True)
    eta_minutes = db.Column(db.Integer, nullable=True)

    # Responder status: responding | arrived
    status = db.Column(db.String(20), nullable=False, default="responding")

    responded_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc)
    )
    arrived_at = db.Column(db.DateTime, nullable=True)

    # Relationship back to SOS
    sos_event = db.relationship("SOSEvent", back_populates="responders")


class VoiceRecord(db.Model):
    __tablename__ = "voice_records"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50), nullable=False)
    audio_file_path = db.Column(db.String(300), nullable=True)
    text_input = db.Column(db.Text, nullable=True)
    input_mode = db.Column(db.String(20), nullable=False)  # "audio" | "text"
    voice_risk = db.Column(db.Boolean, default=False)
    keyword = db.Column(db.String(100), nullable=True)
    confidence = db.Column(db.Float, default=0.0)
    message = db.Column(db.String(200), nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class RiskAnalysis(db.Model):
    __tablename__ = "risk_analyses"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50), nullable=False)
    mode = db.Column(db.String(20), nullable=False)
    input_data = db.Column(db.JSON, nullable=True)
    risk_score = db.Column(db.Integer, nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)
    risk_type = db.Column(db.String(50), nullable=False)
    should_trigger_sos = db.Column(db.Boolean, default=False)
    reasons = db.Column(db.JSON, nullable=True)
    suggestion = db.Column(db.String(300), nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
