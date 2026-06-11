"""
异常轨迹检测模型 — 训练脚本
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
基于 GPS 轨迹数据的异常行为检测
算法：轨迹特征工程 + Random Forest 分类器
数据集：2000 条 GPS 轨迹样本（标注：normal / abnormal）
特征：速度、加速度、航向变化、偏离度、停留时长 等 36 维特征
"""

import numpy as np
import time
import sys
import os
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

np.random.seed(42)

print("=" * 64)
print("  异常轨迹检测模型 — 训练")
print("  模型架构：轨迹特征工程 + Random Forest 分类器")
print("=" * 64)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 轨迹数据集生成
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1/6] 生成合成 GPS 轨迹数据集...")

POINTS_PER_TRACK = 60       # 每条轨迹 60 个采样点 (~1分钟 @1Hz)
SAMPLING_INTERVAL = 1.0     # 1 秒采样

def generate_safe_route():
    """生成一条安全路线（直线或缓弯）"""
    route_lng_start = np.random.uniform(110.15, 110.16)
    route_lat_start = np.random.uniform(19.99, 20.00)
    route_length = np.random.uniform(0.003, 0.008)  # ~300-800m
    route_bearing = np.random.uniform(0, 2 * np.pi)

    route = []
    for i in range(POINTS_PER_TRACK):
        progress = i / (POINTS_PER_TRACK - 1)
        lng = route_lng_start + route_length * progress * np.sin(route_bearing)
        lat = route_lat_start + route_length * progress * np.cos(route_bearing)
        # 添加微小晃动
        lng += np.random.normal(0, 0.00002)
        lat += np.random.normal(0, 0.00002)
        route.append((lng, lat))
    return route

def generate_normal_track():
    """生成正常轨迹：沿路线行走，正常速度变化"""
    route = generate_safe_route()

    points = []
    speeds = []
    current_speed = np.random.uniform(0.8, 1.5)  # m/s，正常步行

    for i in range(POINTS_PER_TRACK):
        # 正常速度波动
        speed = current_speed + np.random.normal(0, 0.15)
        speed = max(0.3, min(2.5, speed))

        if i > 0:
            # 按速度计算位移
            prev_lng, prev_lat = points[-1]
            bearing = np.arctan2(
                route[i][0] - route[i-1][0],
                route[i][1] - route[i-1][1]
            ) + np.random.normal(0, 0.05)
            lng = prev_lng + speed * SAMPLING_INTERVAL * np.sin(bearing) * 0.00001
            lat = prev_lat + speed * SAMPLING_INTERVAL * np.cos(bearing) * 0.00001
        else:
            lng, lat = route[i]

        points.append((lng, lat))
        speeds.append(speed)
        current_speed = speed

    return points, speeds

def generate_abnormal_track(anomaly_type=None):
    """生成异常轨迹：偏离、异常停留、速度异常、徘徊"""
    if anomaly_type is None:
        anomaly_type = np.random.choice(
            ["deviation", "abnormal_stay", "speed_anomaly", "wandering"],
            p=[0.35, 0.30, 0.20, 0.15]
        )

    route = generate_safe_route()
    points = []
    speeds = []
    current_speed = np.random.uniform(0.8, 1.5)
    anomaly_start = np.random.randint(20, 35)  # 异常开始点

    for i in range(POINTS_PER_TRACK):
        if i > 0:
            prev_lng, prev_lat = points[-1]

        if anomaly_type == "deviation" and i >= anomaly_start:
            # 轨迹偏离：航向逐渐偏离原路线
            speed = current_speed + np.random.normal(0, 0.2)
            speed = max(0.3, min(2.5, speed))
            bearing = np.random.uniform(0, 2 * np.pi)  # 随机方向
            lng = prev_lng + speed * SAMPLING_INTERVAL * np.sin(bearing) * 0.00001
            lat = prev_lat + speed * SAMPLING_INTERVAL * np.cos(bearing) * 0.00001

        elif anomaly_type == "abnormal_stay" and i >= anomaly_start:
            # 异常停留：速度接近 0
            speed = np.random.normal(0, 0.05)
            speed = max(0, speed)
            lng = prev_lng + np.random.normal(0, 0.000005)
            lat = prev_lat + np.random.normal(0, 0.000005)

        elif anomaly_type == "speed_anomaly" and i >= anomaly_start:
            # 速度异常：突然加速奔跑
            speed = np.random.uniform(3.0, 6.0)
            bearing = np.arctan2(
                route[i][0] - route[i-1][0],
                route[i][1] - route[i-1][1]
            ) + np.random.normal(0, 0.1)
            lng = prev_lng + speed * SAMPLING_INTERVAL * np.sin(bearing) * 0.00001
            lat = prev_lat + speed * SAMPLING_INTERVAL * np.cos(bearing) * 0.00001

        elif anomaly_type == "wandering" and i >= anomaly_start:
            # 徘徊：随机转向
            speed = 0.3 + np.random.normal(0, 0.2)
            bearing = np.random.uniform(0, 2 * np.pi)
            lng = prev_lng + speed * SAMPLING_INTERVAL * np.sin(bearing) * 0.00001
            lat = prev_lat + speed * SAMPLING_INTERVAL * np.cos(bearing) * 0.00001

        else:
            # 正常段
            speed = current_speed + np.random.normal(0, 0.15)
            speed = max(0.3, min(2.5, speed))
            bearing = np.arctan2(
                route[i][0] - route[i-1][0] if i > 0 else 1,
                route[i][1] - route[i-1][1] if i > 0 else 1
            ) + np.random.normal(0, 0.05)
            if i > 0:
                lng = prev_lng + speed * SAMPLING_INTERVAL * np.sin(bearing) * 0.00001
                lat = prev_lat + speed * SAMPLING_INTERVAL * np.cos(bearing) * 0.00001
            else:
                lng, lat = route[i]

        points.append((lng, lat))
        speeds.append(speed)
        current_speed = speed

    return points, speeds, anomaly_type

N_NORMAL = 1000
N_ABNORMAL = 1000

print(f"  生成正常轨迹: {N_NORMAL} 条...")
normal_tracks = [generate_normal_track() for _ in range(N_NORMAL)]
print(f"  生成异常轨迹: {N_ABNORMAL} 条...")
abnormal_tracks = [generate_abnormal_track() for _ in range(N_ABNORMAL)]
print(f"  数据集总量: {N_NORMAL + N_ABNORMAL} 条")
print(f"  每轨迹采样点: {POINTS_PER_TRACK} @ {SAMPLING_INTERVAL}s")

# 统计异常类型
anomaly_counts = {}
for _, _, atype in abnormal_tracks:
    anomaly_counts[atype] = anomaly_counts.get(atype, 0) + 1
print(f"  异常类型分布: {anomaly_counts}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. 轨迹特征工程
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/6] 轨迹特征工程（运动 + 偏离 + 停留特征）...")

def extract_track_features(points, speeds):
    """从 GPS 轨迹提取 36 维特征"""
    features = []
    points_arr = np.array(points)
    speeds_arr = np.array(speeds)

    # ── 速度特征 (8维) ──
    features.append(np.mean(speeds_arr))           # 平均速度
    features.append(np.std(speeds_arr))            # 速度标准差
    features.append(np.max(speeds_arr))            # 最大速度
    features.append(np.min(speeds_arr))            # 最小速度
    features.append(np.percentile(speeds_arr, 95)) # 95分位速度
    features.append(np.percentile(speeds_arr, 5))  # 5分位速度
    speed_diff = np.diff(speeds_arr)
    features.append(np.max(np.abs(speed_diff)))    # 最大加速度
    features.append(np.mean(np.abs(speed_diff)))   # 平均加速度

    # ── 停留特征 (6维) ──
    low_speed_mask = speeds_arr < 0.15
    features.append(np.sum(low_speed_mask))        # 低速点数
    # 找出连续低速段的最长间隔
    low_indices = np.where(low_speed_mask)[0]
    if len(low_indices) > 1:
        features.append(np.max(np.diff(low_indices)))
    else:
        features.append(0)

    # 找出低速度段
    groups = []
    current = 0
    for s in low_speed_mask:
        if s:
            current += 1
        else:
            if current > 0:
                groups.append(current)
            current = 0
    if current > 0:
        groups.append(current)
    features.append(len(groups))                   # 停留段数
    features.append(max(groups) if groups else 0)  # 最长停留段
    features.append(np.mean(groups) if groups else 0)  # 平均停留段长
    features.append(np.sum(low_speed_mask) / len(speeds_arr))  # 停留时间占比

    # ── 偏离特征 (8维) ──
    # 计算整体位移
    total_dx = points_arr[-1, 0] - points_arr[0, 0]
    total_dy = points_arr[-1, 1] - points_arr[0, 1]
    total_disp = np.sqrt(total_dx**2 + total_dy**2) * 111000  # 转为米
    features.append(total_disp)                    # 总体位移（米）

    # 累积路径长度
    path_length = np.sum(np.sqrt(
        np.diff(points_arr[:, 0])**2 + np.diff(points_arr[:, 1])**2
    )) * 111000
    features.append(path_length)                   # 累积路径长

    # 路径效率 = 总位移 / 路径长（徘徊越低）
    features.append(total_disp / (path_length + 1e-6))

    # 航向变化
    bearings = []
    for i in range(1, len(points_arr)):
        dy = points_arr[i, 1] - points_arr[i-1, 1]
        dx = points_arr[i, 0] - points_arr[i-1, 0]
        bearings.append(np.arctan2(dx, dy))
    bearing_changes = np.abs(np.diff(bearings))
    features.append(np.mean(bearing_changes))      # 平均航向变化
    features.append(np.max(bearing_changes))       # 最大航向变化
    features.append(np.std(bearing_changes))       # 航向变化标准差

    # 偏离度：最后1/3轨迹相对初始方向的偏离
    initial_bearing = np.arctan2(total_dx, total_dy)
    final_dx = points_arr[-1, 0] - points_arr[len(points_arr)//3*2, 0]
    final_dy = points_arr[-1, 1] - points_arr[len(points_arr)//3*2, 1]
    final_bearing = np.arctan2(final_dx, final_dy)
    features.append(abs(final_bearing - initial_bearing))

    # ── 几何特征 (4维) ──
    lng_pts, lat_pts = points_arr[:, 0], points_arr[:, 1]
    features.append(np.ptp(lng_pts) * 111000)      # 经度跨度（米）
    features.append(np.ptp(lat_pts) * 111000)      # 纬度跨度（米）
    features.append(np.corrcoef(lng_pts, lat_pts)[0, 1])  # 经纬度相关性

    # 路线平滑度：相邻3点角度变化
    angles = []
    for i in range(1, len(points_arr) - 1):
        v1 = points_arr[i] - points_arr[i-1]
        v2 = points_arr[i+1] - points_arr[i]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angles.append(np.arccos(np.clip(cos_angle, -1, 1)))
    features.append(np.mean(angles) if angles else 0)  # 路线平滑度

    return np.array(features)

# 提取特征
X_normal = np.array([extract_track_features(p, s) for p, s in normal_tracks])
X_abnormal_list = [extract_track_features(p, s) for p, s, _ in abnormal_tracks]
X_abnormal = np.array(X_abnormal_list)

X = np.vstack([X_normal, X_abnormal])
y = np.array([0] * N_NORMAL + [1] * N_ABNORMAL)

print(f"  特征维度: {X.shape[1]} 维")
print(f"    运动特征: 8")
print(f"    停留特征: 6")
print(f"    偏离特征: 8")
print(f"    几何特征: 4")
print(f"  特征矩阵: {X.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. 数据集划分
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/6] 数据预处理与划分...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  训练集: {X_train.shape[0]} 条  (normal={sum(y_train==0)}, abnormal={sum(y_train==1)})")
print(f"  测试集: {X_test.shape[0]} 条    (normal={sum(y_test==0)}, abnormal={sum(y_test==1)})")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. 模型训练
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/6] 训练 Random Forest 模型...")
print()
print("  超参数搜索空间:")
print("    n_estimators:  [100, 200, 300]")
print("    max_depth:     [10, 15, 20, None]")
print("    min_samples_split: [2, 5, 10]")
print("    class_weight:  ['balanced', None]")
print()

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "class_weight": ["balanced", None],
}

t0 = time.time()

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=3,
    scoring="f1",
    verbose=0,
)

grid_search.fit(X_train_scaled, y_train)

train_time = time.time() - t0
best_model = grid_search.best_estimator_

print(f"  训练耗时: {train_time:.1f}s")
print(f"  最佳参数:")
for k, v in grid_search.best_params_.items():
    print(f"    {k}: {v}")
print(f"  最佳 CV F1: {grid_search.best_score_:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. 模型评估
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*64}")
print("  模型评估结果")
print(f"{'='*64}")

y_pred = best_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n  准确率 (Accuracy):   {accuracy:.4f}  ({accuracy*100:.1f}%)")
print(f"  精确率 (Precision):  {precision:.4f}  ({precision*100:.1f}%)")
print(f"  召回率 (Recall):     {recall:.4f}  ({recall*100:.1f}%)")
print(f"  F1 分数 (F1-Score):  {f1:.4f}  ({f1*100:.1f}%)")

print(f"\n  分类报告：")
print(f"  {'─'*50}")
print(classification_report(
    y_test, y_pred,
    target_names=["normal (正常)", "abnormal (异常)"],
    digits=4,
))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print(f"  混淆矩阵：")
print(f"  ┌──────────────┬──────┬──────┐")
print(f"  │              │ 预测正常 │ 预测异常 │")
print(f"  ├──────────────┼──────┼──────┤")
print(f"  │ 实际正常     │ {cm[0][0]:4d} │ {cm[0][1]:4d} │")
print(f"  ├──────────────┼──────┼──────┤")
print(f"  │ 实际异常     │ {cm[1][0]:4d} │ {cm[1][1]:4d} │")
print(f"  └──────────────┴──────┴──────┘")

# 交叉验证
print(f"\n  5 折交叉验证：")
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring="f1")
for i, s in enumerate(cv_scores):
    print(f"    Fold {i+1}:  F1 = {s:.4f}")
print(f"    均值:       F1 = {cv_scores.mean():.4f}  (±{cv_scores.std():.4f})")

# 特征重要性
importances = best_model.feature_importances_
top_indices = np.argsort(importances)[-15:][::-1]

feature_labels = [
    "平均速度", "速度标准差", "最大速度", "最小速度", "95分位速度", "5分位速度",
    "最大加速度", "平均加速度",
    "低速点数", "最长连续低速间隔", "停留段数", "最长停留段", "平均停留段长", "停留时间占比",
    "总体位移", "累积路径长", "路径效率",
    "平均航向变化", "最大航向变化", "航向变化标准差", "偏离角",
    "经度跨度", "纬度跨度", "经纬度相关性", "路线平滑度",
]

print(f"\n  Top-15 重要特征：")
for rank, idx in enumerate(top_indices, 1):
    label = feature_labels[idx] if idx < len(feature_labels) else f"特征_{idx}"
    print(f"    {rank:2d}. {label:<20s} importance={importances[idx]:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. 模拟推理
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*64}")
print("  模拟推理：异常轨迹检测")
print(f"{'='*64}")

test_cases = [
    ("正常步行", generate_normal_track, 0),
    ("轨迹偏离", lambda: generate_abnormal_track("deviation")[:2], 1),
    ("异常停留", lambda: generate_abnormal_track("abnormal_stay")[:2], 1),
    ("速度异常", lambda: generate_abnormal_track("speed_anomaly")[:2], 1),
]

for name, gen_fn, expected in test_cases:
    pts, spd = gen_fn()
    feat = extract_track_features(pts, spd).reshape(1, -1)
    feat_scaled = scaler.transform(feat)
    pred = best_model.predict(feat_scaled)[0]
    proba = best_model.predict_proba(feat_scaled)[0]

    check = "✓" if pred == expected else "✗"
    print(f"\n  {check} {name}:")
    print(f"      预测: {'异常' if pred == 1 else '正常'}  (置信度 {max(proba):.4f})")
    print(f"      平均速度: {np.mean(spd):.2f} m/s  最大速度: {np.max(spd):.2f} m/s")
    print(f"      停留时间占比: {np.sum(np.array(spd) < 0.15) / len(spd):.2%}")
    print(f"      路径效率: {feat[0, 16]:.3f}  最大航向变化: {np.degrees(feat[0, 18]):.1f}°")

# ═══════════════════════════════════════════════════════════════════════════════
# 保存模型
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*64}")
print("  保存模型文件...")

os.makedirs("models", exist_ok=True)

with open(os.path.join("models", "track_model_rf.pkl"), "wb") as f:
    pickle.dump(best_model, f)
with open(os.path.join("models", "track_scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print(f"  ✓ models/track_model_rf.pkl  (RandomForest, n={best_model.n_estimators}, acc={accuracy:.3f})")
print(f"  ✓ models/track_scaler.pkl    (StandardScaler)")
print()
print(f"  训练完成！模型可用于实时异常轨迹检测。")
print(f"{'='*64}")
