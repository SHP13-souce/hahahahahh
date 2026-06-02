from flask import Blueprint, request, jsonify
from models import VoiceRecord, db
from services import voice_service

voice_bp = Blueprint("voice", __name__)


@voice_bp.route("/api/voice/detect", methods=["POST"])
def detect_voice():
    """Voice detection endpoint.

    Accepts:
      - multipart/form-data: userId + audioFile
      - application/json:     { userId, text }
    """
    # --- Text mode (JSON) ---
    if request.is_json:
        data = request.get_json() or {}
        user_id = data.get("userId", "")
        text = data.get("text", "")

        if not user_id:
            return jsonify({"success": False, "message": "userId 不能为空"}), 400

        result = voice_service.detect_text_distress(text)

        # Persist record
        record = VoiceRecord(
            user_id=user_id,
            text_input=text,
            input_mode="text",
            voice_risk=result["voiceRisk"],
            keyword=result["keyword"],
            confidence=result["confidence"],
            message=result["message"],
        )
        db.session.add(record)
        db.session.commit()

        return jsonify({"success": True, **result})

    # --- Audio file upload mode ---
    if "audioFile" in request.files:
        user_id = request.form.get("userId", "")
        audio_file = request.files["audioFile"]

        if not user_id:
            return jsonify({"success": False, "message": "userId 不能为空"}), 400

        if not audio_file or audio_file.filename == "":
            return jsonify({"success": False, "message": "未上传音频文件"}), 400

        # Save audio file
        file_path = voice_service.save_audio_file(audio_file)

        # TODO: Replace with real ASR → distress detection pipeline
        # For now, returns a safe placeholder result
        result = {
            "voiceRisk": False,
            "keyword": "",
            "confidence": 0.15,
            "message": "音频已接收，待语音识别处理",
        }

        # Persist record
        record = VoiceRecord(
            user_id=user_id,
            audio_file_path=file_path,
            input_mode="audio",
            voice_risk=result["voiceRisk"],
            keyword=result["keyword"],
            confidence=result["confidence"],
            message=result["message"],
        )
        db.session.add(record)
        db.session.commit()

        return jsonify({"success": True, **result})

    # Neither JSON nor file upload
    return jsonify({"success": False, "message": "请提供 text 字段或上传 audioFile"}), 400
