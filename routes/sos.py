import math
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify
from models import SOSEvent, Responder, db

sos_bp = Blueprint("sos", __name__)

# Statuses considered "active" (visible to volunteers in polling)
ACTIVE_STATUSES = ["pending", "responding"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _generate_sos_id() -> str:
    """Generate a unique sequential SOS ID: sos_001, sos_002, ..."""
    count = SOSEvent.query.count()
    return f"sos_{count + 1:03d}"


def _haversine_distance(lng1: float, lat1: float, lng2: float, lat2: float) -> float:
    """Calculate distance in meters between two GPS coordinates (Haversine)."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _estimate_eta_minutes(
    vol_lng: float, vol_lat: float, sos_lng: float, sos_lat: float
) -> int:
    """Estimate ETA in minutes based on walking distance."""
    distance_m = _haversine_distance(vol_lng, vol_lat, sos_lng, sos_lat)
    if distance_m < 100:
        return 1
    elif distance_m < 300:
        return 2
    elif distance_m < 500:
        return 3
    elif distance_m < 1000:
        return 5
    else:
        return max(1, min(60, int(distance_m / 150)))  # ~150 m/min walking


def _sync_sos_status(sos: SOSEvent) -> None:
    """Keep SOS status in sync with its responders.

    - No responders → pending
    - At least one responder with status=responding, none arrived → responding
    - At least one responder arrived → arrived
    """
    if not sos.responders:
        return  # stays as-is (typically 'pending')

    arrived = [r for r in sos.responders if r.status == "arrived"]
    responding = [r for r in sos.responders if r.status == "responding"]

    if arrived:
        sos.status = "arrived"
    elif responding:
        sos.status = "responding"


# ── Routes ───────────────────────────────────────────────────────────────────

@sos_bp.route("/api/sos/create", methods=["POST"])
def create_sos():
    """Create a new SOS event. Called by requester after countdown expires."""
    data = request.json or {}

    location = data.get("location", {})
    risk = data.get("risk", {})

    sos = SOSEvent(
        sos_id=_generate_sos_id(),
        user_id=data.get("userId", ""),
        user_name=data.get("userName", ""),
        location_lng=location.get("lng", 0),
        location_lat=location.get("lat", 0),
        location_address=location.get("address", ""),
        risk_score=risk.get("riskScore", 0),
        risk_level=risk.get("riskLevel", "low"),
        risk_type=risk.get("riskType", "multi_source_abnormal"),
        risk_reasons=[{"label": r} for r in risk.get("reasons", [])],
        risk_suggestion=risk.get("suggestion", ""),
        source=data.get("source", "auto_behavior_model"),
        status="pending",
    )
    db.session.add(sos)
    db.session.commit()

    return jsonify(
        {
            "success": True,
            "sosId": sos.sos_id,
            "status": sos.status,
            "message": "SOS 已创建，正在通知附近志愿者",
        }
    )


@sos_bp.route("/api/sos/active", methods=["GET"])
def get_active_sos():
    """Return the most recent active SOS with all responders.

    Polled by volunteer app every 2 seconds.
    """
    active = (
        SOSEvent.query.filter(SOSEvent.status.in_(ACTIVE_STATUSES))
        .order_by(SOSEvent.created_at.desc())
        .first()
    )

    if not active:
        return jsonify({"success": True, "hasActiveSOS": False, "data": None})

    return jsonify({"success": True, "hasActiveSOS": True, "data": active.to_dict()})


@sos_bp.route("/api/sos/respond", methods=["POST"])
def respond_sos():
    """Volunteer accepts a SOS. Creates a Responder record.

    Multiple volunteers can respond to the same SOS.
    A volunteer cannot respond twice to the same SOS.
    """
    data = request.json or {}
    sos_id = data.get("sosId", "")
    volunteer_id = data.get("volunteerId", "")

    sos = SOSEvent.query.filter_by(sos_id=sos_id).first()
    if not sos:
        return jsonify({"success": False, "message": "SOS 事件不存在"}), 404

    if sos.status not in ACTIVE_STATUSES:
        return jsonify(
            {
                "success": False,
                "message": f"SOS 当前状态为 {sos.status}，无法响应",
            }
        ), 400

    # Prevent duplicate response from same volunteer
    existing = Responder.query.filter_by(
        sos_id=sos_id, volunteer_id=volunteer_id
    ).first()
    if existing:
        return jsonify(
            {
                "success": False,
                "message": "您已响应过此 SOS，请勿重复响应",
            }
        ), 400

    vol_lng = data.get("location", {}).get("lng", 0)
    vol_lat = data.get("location", {}).get("lat", 0)
    eta = _estimate_eta_minutes(vol_lng, vol_lat, sos.location_lng, sos.location_lat)

    # Create responder record
    responder = Responder(
        sos_id=sos_id,
        volunteer_id=volunteer_id,
        volunteer_name=data.get("volunteerName", ""),
        volunteer_type=data.get("volunteerType", ""),
        phone=data.get("phone", ""),
        location_lng=vol_lng,
        location_lat=vol_lat,
        eta_minutes=eta,
        status="responding",
    )
    db.session.add(responder)

    # Bump SOS status if it was pending
    if sos.status == "pending":
        sos.status = "responding"

    db.session.commit()

    return jsonify(
        {
            "success": True,
            "status": sos.status,
            "message": "志愿者已响应，正在前往",
            "responder": {
                "volunteerId": responder.volunteer_id,
                "volunteerName": responder.volunteer_name,
                "volunteerType": responder.volunteer_type,
                "phone": responder.phone,
                "etaMinutes": eta,
            },
        }
    )


@sos_bp.route("/api/sos/arrive", methods=["POST"])
def arrive_sos():
    """Volunteer confirms arrival. Updates the Responder record.

    First volunteer to arrive sets SOS status → arrived.
    Subsequent volunteers can still arrive (their Responder is updated).
    """
    data = request.json or {}
    sos_id = data.get("sosId", "")
    volunteer_id = data.get("volunteerId", "")

    sos = SOSEvent.query.filter_by(sos_id=sos_id).first()
    if not sos:
        return jsonify({"success": False, "message": "SOS 事件不存在"}), 404

    # Find this volunteer's responder record
    responder = Responder.query.filter_by(
        sos_id=sos_id, volunteer_id=volunteer_id
    ).first()
    if not responder:
        return jsonify(
            {
                "success": False,
                "message": "未找到该志愿者的响应记录",
            }
        ), 404

    if responder.status == "arrived":
        return jsonify(
            {
                "success": False,
                "message": "您已确认过到达",
            }
        ), 400

    if sos.status == "cancelled" or sos.status == "closed":
        return jsonify(
            {
                "success": False,
                "message": f"SOS 当前状态为 {sos.status}，无法确认到达",
            }
        ), 400

    # Mark this responder as arrived
    responder.status = "arrived"
    responder.arrived_at = datetime.now(timezone.utc)

    # Sync SOS overall status
    _sync_sos_status(sos)

    db.session.commit()

    return jsonify(
        {
            "success": True,
            "status": sos.status,
            "message": "志愿者已到达，求助流程完成",
        }
    )
