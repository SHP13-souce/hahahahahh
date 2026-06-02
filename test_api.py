"""
End-to-end test: full SOS data flow with multi-volunteer support.
Run after starting the server:  python app.py
"""
import requests
import json
import sys
import os

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://127.0.0.1:5000"

passed = 0
failed = 0


def test(label, expected_status, fn):
    global passed, failed
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    try:
        resp = fn()
        print(f"  Status: {resp.status_code}")
        body = resp.json()
        body_str = json.dumps(body, ensure_ascii=False, indent=2)
        if len(body_str) > 2500:
            body_str = body_str[:2500] + "\n  ... (truncated)"
        print(f"  Body: {body_str}")

        if resp.status_code == expected_status:
            print(f"  [PASS]")
            passed += 1
            return body
        else:
            print(f"  [FAIL] expected {expected_status}, got {resp.status_code}")
            failed += 1
            return None
    except Exception as e:
        print(f"  [ERROR] {e}")
        failed += 1
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Voice & Risk
# ═══════════════════════════════════════════════════════════════════════════════

# Test 1: Voice — text distress
test(
    "POST /api/voice/detect [text mode - distress]",
    200,
    lambda: requests.post(
        f"{BASE}/api/voice/detect",
        json={"userId": "u001", "text": "救命！有人吗！"},
    ),
)

# Test 2: Voice — normal text
test(
    "POST /api/voice/detect [text mode - normal]",
    200,
    lambda: requests.post(
        f"{BASE}/api/voice/detect",
        json={"userId": "u001", "text": "今天天气不错"},
    ),
)

# Test 3: Voice — audio upload
test(
    "POST /api/voice/detect [audio file]",
    200,
    lambda: requests.post(
        f"{BASE}/api/voice/detect",
        data={"userId": "u001"},
        files={"audioFile": ("test.wav", b"fake-audio", "audio/wav")},
    ),
)

# Test 4: Risk — high risk
test(
    "POST /api/risk/analyze [high risk]",
    200,
    lambda: requests.post(
        f"{BASE}/api/risk/analyze",
        json={
            "userId": "u001",
            "mode": "demo_replay",
            "location": {"lng": 110.152305, "lat": 19.99839},
            "scene": {
                "timePeriod": "night",
                "areaType": "urban_village",
                "inRiskArea": True,
                "volunteerCount": 0,
            },
            "track": {"staySeconds": 12, "speed": 0.1, "deviation": True},
            "sensor": {"fallDetected": True, "shakeLevel": "high", "stillSeconds": 10},
            "voice": {"voiceRisk": True, "keyword": "救命", "confidence": 0.92},
        },
    ),
)

# Test 5: Risk — low risk
test(
    "POST /api/risk/analyze [low risk]",
    200,
    lambda: requests.post(
        f"{BASE}/api/risk/analyze",
        json={
            "userId": "u002",
            "mode": "live",
            "location": {"lng": 110.152, "lat": 19.99},
            "scene": {"timePeriod": "day", "areaType": "normal", "inRiskArea": False, "volunteerCount": 3},
            "track": {"staySeconds": 2, "speed": 1.2, "deviation": False},
            "sensor": {"fallDetected": False, "shakeLevel": "normal", "stillSeconds": 1},
            "voice": {"voiceRisk": False, "keyword": "", "confidence": 0.10},
        },
    ),
)

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: SOS lifecycle — multi-volunteer
# ═══════════════════════════════════════════════════════════════════════════════

# Test 6: Create SOS
r6 = test(
    "POST /api/sos/create",
    200,
    lambda: requests.post(
        f"{BASE}/api/sos/create",
        json={
            "userId": "u001",
            "userName": "沈诗雨",
            "location": {"lng": 110.152305, "lat": 19.99839, "address": "白沙门村偏僻路段"},
            "risk": {
                "riskScore": 92,
                "riskLevel": "high",
                "riskType": "multi_source_abnormal",
                "reasons": ["识别到疑似求救语音", "处于夜间城中村风险场景"],
                "suggestion": "建议附近志愿者响应，优先双人同行",
            },
            "source": "auto_behavior_model",
        },
    ),
)

# Test 7: Active SOS (pending, 0 responders)
test(
    "GET /api/sos/active [pending, 0 responders]",
    200,
    lambda: requests.get(f"{BASE}/api/sos/active"),
)

# Test 8: Volunteer 1 responds
r8 = test(
    "POST /api/sos/respond [v001 — 王秀兰]",
    200,
    lambda: requests.post(
        f"{BASE}/api/sos/respond",
        json={
            "sosId": r6["sosId"] if r6 else "",
            "volunteerId": "v001",
            "volunteerName": "王秀兰",
            "volunteerType": "community_guard",
            "phone": "13800000000",
            "location": {"lng": 110.1525, "lat": 19.9986},
        },
    ),
)

# Test 9: Active SOS (responding, 1 responder)
test(
    "GET /api/sos/active [responding, 1 responder]",
    200,
    lambda: requests.get(f"{BASE}/api/sos/active"),
)

# Test 10: Volunteer 2 responds (multi-volunteer!)
r10 = test(
    "POST /api/sos/respond [v002 — 张三 — multi-volunteer]",
    200,
    lambda: requests.post(
        f"{BASE}/api/sos/respond",
        json={
            "sosId": r6["sosId"] if r6 else "",
            "volunteerId": "v002",
            "volunteerName": "张三",
            "volunteerType": "community_guard",
            "phone": "13900000000",
            "location": {"lng": 110.1530, "lat": 19.9990},
        },
    ),
)

# Test 11: Active SOS (responding, 2 responders)
test(
    "GET /api/sos/active [responding, 2 responders]",
    200,
    lambda: requests.get(f"{BASE}/api/sos/active"),
)

# Test 12: Volunteer 1 tries duplicate respond → should fail
test(
    "POST /api/sos/respond [v001 duplicate — should fail]",
    400,
    lambda: requests.post(
        f"{BASE}/api/sos/respond",
        json={
            "sosId": r6["sosId"] if r6 else "",
            "volunteerId": "v001",
            "volunteerName": "王秀兰",
            "volunteerType": "community_guard",
            "phone": "13800000000",
            "location": {"lng": 110.1525, "lat": 19.9986},
        },
    ),
)

# Test 13: Volunteer 1 arrives
test(
    "POST /api/sos/arrive [v001 arrives]",
    200,
    lambda: requests.post(
        f"{BASE}/api/sos/arrive",
        json={"sosId": r6["sosId"] if r6 else "", "volunteerId": "v001"},
    ),
)

# Test 14: Active SOS after first arrival — still active (v002 has not arrived)
test(
    "GET /api/sos/active [v001 arrived, v002 still responding — should be active]",
    200,
    lambda: requests.get(f"{BASE}/api/sos/active"),
)

# Test 15: Volunteer 2 also arrives
test(
    "POST /api/sos/arrive [v002 arrives — all arrived]",
    200,
    lambda: requests.post(
        f"{BASE}/api/sos/arrive",
        json={"sosId": r6["sosId"] if r6 else "", "volunteerId": "v002"},
    ),
)

# Test 16: Active SOS after all arrived — should be inactive
test(
    "GET /api/sos/active [all arrived — no active]",
    200,
    lambda: requests.get(f"{BASE}/api/sos/active"),
)

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Edge cases
# ═══════════════════════════════════════════════════════════════════════════════

# Create a fresh SOS for edge case testing
r_ec = test(
    "POST /api/sos/create [setup for edge cases]",
    200,
    lambda: requests.post(
        f"{BASE}/api/sos/create",
        json={
            "userId": "u003",
            "userName": "测试用户",
            "location": {"lng": 110.152, "lat": 19.998, "address": "测试地址"},
            "risk": {
                "riskScore": 75,
                "riskLevel": "high",
                "riskType": "multi_source_abnormal",
                "reasons": ["测试"],
                "suggestion": "测试",
            },
            "source": "auto_behavior_model",
        },
    ),
)

# Edge case: volunteer tries to arrive without responding first
test(
    "POST /api/sos/arrive [no prior respond — should fail]",
    404,
    lambda: requests.post(
        f"{BASE}/api/sos/arrive",
        json={"sosId": r_ec["sosId"] if r_ec else "", "volunteerId": "v999"},
    ),
)

# Edge case: respond to non-existent SOS
test(
    "POST /api/sos/respond [non-existent SOS — should fail]",
    404,
    lambda: requests.post(
        f"{BASE}/api/sos/respond",
        json={
            "sosId": "sos_nonexistent",
            "volunteerId": "v001",
            "volunteerName": "测试",
            "volunteerType": "test",
            "phone": "000",
            "location": {"lng": 0, "lat": 0},
        },
    ),
)

print(f"\n{'='*60}")
print(f"  Results: {passed} passed, {failed} failed out of {passed + failed}")
print(f"{'='*60}")
