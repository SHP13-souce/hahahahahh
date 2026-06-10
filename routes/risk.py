from flask import Blueprint, request, jsonify
from models import RiskAnalysis, db

risk_bp = Blueprint("risk", __name__)


@risk_bp.route("/api/risk/analyze", methods=["POST"])
def analyze_risk():
    """Multi-source risk scoring endpoint.

    Computes a risk score (0–100) from scene, track, sensor, and voice inputs,
    then maps it to a risk level and decides whether to trigger an SOS.
    """
    data = request.json or {}

    score = 0
    reasons = []

    scene = data.get("scene", {})
    track = data.get("track", {})
    sensor = data.get("sensor", {})
    voice = data.get("voice", {})

    # ── Voice: distress keywords detected ──
    if voice.get("voiceRisk"):
        score += 35
        reasons.append(
            {
                "type": "voice",
                "label": "识别到疑似求救语音",
                "detail": f"关键词：{voice.get('keyword', '')}，置信度：{int(voice.get('confidence', 0) * 100)}%",
                "score": 35,
            }
        )

    # ── Sensor: fall detected ──
    if sensor.get("fallDetected"):
        score += 35
        reasons.append(
            {
                "type": "sensor",
                "label": "检测到疑似摔倒行为",
                "detail": "出现剧烈冲击、姿态变化或持续静止",
                "score": 35,
            }
        )

    # ── Scene: nighttime ──
    if scene.get("timePeriod") == "night":
        score += 15
        reasons.append(
            {
                "type": "scene",
                "label": "处于夜间出行时段",
                "detail": "夜间场景风险升高",
                "score": 15,
            }
        )

    # ── Scene: in risk area ──
    if scene.get("inRiskArea"):
        score += 20
        reasons.append(
            {
                "type": "scene",
                "label": "进入高风险区域",
                "detail": "用户进入城中村偏僻风险区域",
                "score": 20,
            }
        )

    # ── Scene: urban village / village ──
    if scene.get("areaType") in ["urban_village", "village"]:
        score += 10
        reasons.append(
            {
                "type": "scene",
                "label": "处于城中村/乡村复杂环境",
                "detail": "周边环境复杂，夜间风险更高",
                "score": 10,
            }
        )

    # ── Track: abnormal stay (staySeconds ≥ 8 AND speed ≤ 0.2) ──
    if track.get("staySeconds", 0) >= 8 and track.get("speed", 1) <= 0.2:
        score += 20
        reasons.append(
            {
                "type": "track",
                "label": "检测到异常停留",
                "detail": f"用户停留 {track.get('staySeconds')} 秒，速度接近 0",
                "score": 20,
            }
        )

    # ── Track: route deviation ──
    if track.get("deviation"):
        score += 15
        reasons.append(
            {
                "type": "track",
                "label": "检测到轨迹偏离",
                "detail": "当前路线偏离常规安全路线",
                "score": 15,
            }
        )

    # ── Sensor: prolonged stillness after fall (stillSeconds ≥ 5) ──
    if sensor.get("stillSeconds", 0) >= 5:
        score += 15
        reasons.append(
            {
                "type": "sensor",
                "label": "摔倒后持续静止",
                "detail": f"用户持续静止 {sensor.get('stillSeconds')} 秒",
                "score": 15,
            }
        )

    # ── Volunteer coverage: volunteerCount ≤ 0 ──
    if scene.get("volunteerCount", 1) <= 0:
        score += 10
        reasons.append(
            {
                "type": "volunteer",
                "label": "附近志愿者覆盖不足",
                "detail": "当前附近在线志愿者数量不足",
                "score": 10,
            }
        )

    # ── Cap score at 100 ──
    score = min(score, 100)

    # ── Determine risk level & trigger ──
    if score >= 70:
        risk_level = "high"
        should_trigger = True
        suggestion = "启动自动 SOS 倒计时"
    elif score >= 40:
        risk_level = "medium"
        should_trigger = False
        suggestion = "建议开启陪伴守护并提醒用户确认安全"
    else:
        risk_level = "low"
        should_trigger = False
        suggestion = "继续保持普通守护"

    # ── Persist analysis record ──
    analysis = RiskAnalysis(
        user_id=data.get("userId", ""),
        mode=data.get("mode", "live"),
        input_data=data,
        risk_score=score,
        risk_level=risk_level,
        risk_type="multi_source_abnormal",
        should_trigger_sos=should_trigger,
        reasons=reasons,
        suggestion=suggestion,
    )
    db.session.add(analysis)
    db.session.commit()

    return jsonify(
        {
            "success": True,
            "riskScore": score,
            "riskLevel": risk_level,
            "riskType": "multi_source_abnormal",
            "shouldTriggerSOS": should_trigger,
            "countdownSeconds": 10,
            "reasons": reasons,
            "suggestion": suggestion,
        }
    )
