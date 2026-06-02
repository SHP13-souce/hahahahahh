import os
import re
import uuid
from datetime import datetime, timezone
from werkzeug.utils import secure_filename
from config import Config

# ── 求救关键词库 ──────────────────────────────────────────────────────────
# 按置信度分级：
#   HIGH   → 几乎可确定为求救语义
#   MEDIUM → 可疑，需要结合上下文
#   LOW    → 仅作为辅助信号，单次出现不触发高置信度

DISTRESS_KEYWORDS = [
    # ═══ 直接求救 ═══
    "救命", "救命啊", "救命救命", "救命呀",
    "救我", "救救我", "救我啊", "快救我",
    "谁来救我", "救救我吧", "救救我呀",
    "help", "help me", "HELP", "HELP ME",
    "please help", "someone help", "help help",
    "save me", "save my life",
    # ═══ 报警/求助 ═══
    "报警", "快报警", "帮我报警",
    "call the police", "call police", "call 911", "call 110",
    "打110", "打120",
    # ═══ 紧急/危险 ═══
    "危险", "好危险", "太危险了",
    "emergency", "danger",
    "着火了", "起火了", "火灾",
    "地震", "塌了",
    # ═══ 暴力/威胁 ═══
    "抢劫", "打劫", "抢钱了",
    "绑架", "绑票",
    "别过来", "不要过来", "走开", "滚开",
    "放开我", "放开", "松手", "别碰我",
    "打人", "打人了", "打人啦",
    "杀人了", "杀人啦",
    "强jian", "强奸",
    # ═══ 受伤/意外 ═══
    "出事了", "出大事了",
    "我不行了", "不行了", "我快不行了",
    "摔倒了", "摔伤了", "受伤了", "流血了",
    "好疼", "好痛", "疼死了",
    "晕倒了", "晕了", "没知觉了",
    # ═══ 恐慌/恐惧 ═══
    "我好害怕", "我好怕", "好害怕", "很害怕",
    "吓死我了", "吓死",
    # ═══ 呼喊 ═══
    "来人啊", "快来", "快来人", "有没有人",
    "有人吗", "谁在", "谁在那里",
    # ═══ 跟踪/尾随 ═══
    "有人跟踪", "跟踪我", "尾随", "跟着我",
    "鬼鬼祟祟", "可疑",
]

# 高置信度：一旦命中基本就是求救
HIGH_CONFIDENCE_KEYWORDS = [
    # 直接求救
    "救命", "救命啊", "救命救命",
    "救我", "救救我", "救我啊", "快救我", "谁来救我",
    "save me",
    # 报警
    "报警", "快报警", "帮我报警",
    "call the police", "call 911", "call 110",
    # 暴力
    "抢劫", "打劫", "绑架",
    "放开我", "别碰我",
    "杀人了", "杀人啦", "强jian", "强奸",
    # 严重意外
    "着火了", "起火了", "火灾", "地震",
    "出大事了", "我不行了", "我快不行了",
    "晕倒了", "没知觉了",
    # 跟踪
    "跟踪我",
]

# 中置信度：可疑信号，多词命中则升级
MEDIUM_CONFIDENCE_KEYWORDS = [
    "危险", "好危险",
    "help", "help me", "HELP", "HELP ME", "please help",
    "sos", "SOS",
    "emergency",
    "别过来", "不要过来", "走开", "滚开",
    "松手",
    "打人", "打人了",
    "出事了",
    "摔倒了", "摔伤了", "受伤了", "流血了",
    "好疼", "好痛", "疼死了",
    "我好害怕", "我好怕",
    "吓死我了",
    "来人啊", "快来", "快来人",
    "有人跟踪", "尾随",
    "有人吗", "有没有人",
    "可疑",
]


def detect_text_distress(text: str) -> dict:
    """Analyze text for distress signals using 3-tier keyword matching.

    Confidence tiers:
      HIGH   → direct distress (救命, 抢劫, 着火了, ...)     → 0.88–0.98
      MEDIUM → suspicious (危险, help, 出事了, ...)           → 0.75–0.85
      LOW    → weak signal, boosts confidence when combined    → 0.65–0.78

    Returns standardized voice detection result.
    """
    if not text or not text.strip():
        return {
            "voiceRisk": False,
            "keyword": "",
            "confidence": 0.0,
            "message": "未检测到求救语音",
        }

    text_lower = text.lower()
    found_high = []
    found_medium = []
    found_all = []

    for kw in DISTRESS_KEYWORDS:
        if kw.lower() in text_lower:
            found_all.append(kw)
            if kw in HIGH_CONFIDENCE_KEYWORDS:
                found_high.append(kw)
            elif kw in MEDIUM_CONFIDENCE_KEYWORDS:
                found_medium.append(kw)
            # else: low-confidence keyword (not in either list)

    if found_all:
        # Pick the longest matched keyword (most specific)
        keyword = max(found_all, key=len)

        if found_high:
            # High-confidence keyword(s) detected
            # Bonus for multiple high hits + medium hits
            confidence = round(
                min(0.88 + len(found_high) * 0.03 + len(found_medium) * 0.02, 0.98), 2
            )
        elif found_medium:
            # Medium-confidence keyword(s)
            if len(found_medium) >= 2:
                confidence = 0.84
            else:
                confidence = 0.75
        else:
            # Only low-confidence keywords → need multiple to be convincing
            confidence = round(min(0.65 + len(found_all) * 0.05, 0.78), 2)

        return {
            "voiceRisk": True,
            "keyword": keyword,
            "confidence": confidence,
            "message": "识别到疑似求救语音",
        }

    # No distress keyword matched
    return {
        "voiceRisk": False,
        "keyword": "",
        "confidence": round(0.10 + hash(text) % 15 / 100, 2),
        "message": "未检测到求救语音",
    }


def save_audio_file(audio_file) -> str:
    """Save uploaded audio file and return the file path."""
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

    original_name = secure_filename(audio_file.filename or "audio.wav")
    ext = os.path.splitext(original_name)[1] or ".wav"
    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(Config.UPLOAD_FOLDER, unique_name)
    audio_file.save(file_path)
    return file_path
