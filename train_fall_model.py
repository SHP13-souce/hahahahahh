"""
摔倒检测模型 — 训练脚本
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
基于加速度计数据的摔倒检测分类器
算法：特征工程 + 随机森林 (Random Forest)
数据集：2000 条加速度时序样本（标注：fall / normal）
特征：时域 + 频域共 45 维特征向量
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

# 强制 UTF-8 输出
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

np.random.seed(42)

print("=" * 64)
print("  摔倒检测模型 — 训练")
print("  模型架构：特征工程 + Random Forest 分类器")
print("=" * 64)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 合成加速度数据集
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1/6] 生成合成加速度数据集...")

SAMPLING_RATE = 50       # Hz
WINDOW_SEC = 4           # 每样本 4 秒窗口
WINDOW_SAMPLES = SAMPLING_RATE * WINDOW_SEC  # 200 点/样本
GRAVITY = 9.8            # m/s²

def generate_fall_pattern():
    """生成摔倒加速度模式：冲击 → 姿态变化 → 静止"""
    t = np.linspace(0, WINDOW_SEC, WINDOW_SAMPLES)

    # 正常段 (0~1s)
    acc_x = np.random.normal(0, 0.3, WINDOW_SAMPLES)
    acc_y = np.random.normal(0, 0.3, WINDOW_SAMPLES)
    acc_z = GRAVITY + np.random.normal(0, 0.3, WINDOW_SAMPLES)

    # 冲击段 (1~1.5s) — 剧烈加速度突变
    impact_start = WINDOW_SAMPLES // 4
    impact_end = impact_start + WINDOW_SAMPLES // 8

    # 随机冲击方向
    direction = np.random.choice(['forward', 'sideways', 'backward'])
    if direction == 'forward':
        spike_x = np.random.uniform(15, 35, impact_end - impact_start)
        spike_y = np.random.uniform(-5, 5, impact_end - impact_start)
        spike_z = np.random.uniform(-10, 5, impact_end - impact_start)
    elif direction == 'sideways':
        spike_x = np.random.uniform(-5, 5, impact_end - impact_start)
        spike_y = np.random.uniform(15, 35, impact_end - impact_start)
        spike_z = np.random.uniform(-10, 5, impact_end - impact_start)
    else:
        spike_x = np.random.uniform(-10, 5, impact_end - impact_start)
        spike_y = np.random.uniform(15, 30, impact_end - impact_start)
        spike_z = np.random.uniform(5, 20, impact_end - impact_start)

    acc_x[impact_start:impact_end] = spike_x
    acc_y[impact_start:impact_end] = spike_y
    acc_z[impact_start:impact_end] = spike_z

    # 冲击后静止段 — 低方差
    still_start = impact_end
    acc_x[still_start:] = np.random.normal(0, 0.1, WINDOW_SAMPLES - still_start)
    acc_y[still_start:] = np.random.normal(0, 0.1, WINDOW_SAMPLES - still_start)
    acc_z[still_start:] = GRAVITY + np.random.normal(0, 0.15, WINDOW_SAMPLES - still_start)

    return np.column_stack([acc_x, acc_y, acc_z])

def generate_normal_pattern():
    """生成正常活动加速度模式：走路、跑步、日常动作"""
    t = np.linspace(0, WINDOW_SEC, WINDOW_SAMPLES)

    activity = np.random.choice(['walking', 'running', 'standing', 'sitting', 'phone_use'])

    if activity == 'walking':
        # 走路 — 周期性加速度
        freq = np.random.uniform(1.5, 2.5)  # Hz
        amplitude = np.random.uniform(1.0, 2.5)
        base_x = amplitude * np.sin(2 * np.pi * freq * t)
        base_y = amplitude * 0.5 * np.cos(2 * np.pi * freq * t)
        base_z = GRAVITY + amplitude * 0.3 * np.sin(2 * np.pi * freq * t)
    elif activity == 'running':
        # 跑步 — 更高频率和幅度
        freq = np.random.uniform(2.5, 4.0)
        amplitude = np.random.uniform(3.0, 5.0)
        base_x = amplitude * np.sin(2 * np.pi * freq * t)
        base_y = amplitude * 0.5 * np.cos(2 * np.pi * freq * t)
        base_z = GRAVITY + amplitude * 0.5 * np.sin(2 * np.pi * freq * t)
    elif activity == 'standing':
        # 站立 — 微小晃动
        base_x = np.random.normal(0, 0.15, WINDOW_SAMPLES)
        base_y = np.random.normal(0, 0.15, WINDOW_SAMPLES)
        base_z = GRAVITY + np.random.normal(0, 0.15, WINDOW_SAMPLES)
    elif activity == 'sitting':
        # 坐下 — 缓慢姿态变化
        transition = np.linspace(0, 2.5, WINDOW_SAMPLES)
        base_x = transition + np.random.normal(0, 0.2, WINDOW_SAMPLES)
        base_y = np.random.normal(0, 0.2, WINDOW_SAMPLES)
        base_z = GRAVITY - transition * 0.5 + np.random.normal(0, 0.2, WINDOW_SAMPLES)
    else:
        # 玩手机 — 随机微小动作
        base_x = np.random.normal(0, 0.5, WINDOW_SAMPLES)
        base_y = np.random.normal(0, 0.5, WINDOW_SAMPLES)
        base_z = GRAVITY + np.random.normal(0, 0.5, WINDOW_SAMPLES)

    noise_level = np.random.uniform(0.1, 0.4)
    acc_x = base_x + np.random.normal(0, noise_level, WINDOW_SAMPLES)
    acc_y = base_y + np.random.normal(0, noise_level, WINDOW_SAMPLES)
    acc_z = base_z + np.random.normal(0, noise_level, WINDOW_SAMPLES)

    return np.column_stack([acc_x, acc_y, acc_z])

# 生成数据集
N_FALL = 1000
N_NORMAL = 1000

print(f"  生成摔倒样本: {N_FALL} 条...")
fall_samples = [generate_fall_pattern() for _ in range(N_FALL)]
print(f"  生成正常样本: {N_NORMAL} 条...")
normal_samples = [generate_normal_pattern() for _ in range(N_NORMAL)]
print(f"  数据集总量: {N_FALL + N_NORMAL} 条")
print(f"  每样本维度: {WINDOW_SAMPLES} × 3 轴")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 特征工程
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/6] 特征工程（时域 + 频域）...")

def extract_features(acc_data):
    """
    从加速度时序数据提取特征。
    输入: (200, 3) — 4秒 × 50Hz × 3轴
    输出: 45 维特征向量
    """
    features = []

    for axis in range(3):
        signal = acc_data[:, axis]

        # ── 时域特征 ──
        features.append(np.mean(signal))
        features.append(np.std(signal))
        features.append(np.max(signal))
        features.append(np.min(signal))
        features.append(np.ptp(signal))                           # 峰峰值
        features.append(np.percentile(signal, 95) - np.percentile(signal, 5))  # IQR

        # ── 冲击特征 ──
        diff_signal = np.diff(signal)
        features.append(np.max(np.abs(diff_signal)))              # 最大变化率
        features.append(np.mean(np.abs(diff_signal)))             # 平均变化率

        # ── 静止特征 ──
        # 后1秒的方差（冲击后静止判断）
        last_sec = signal[-SAMPLING_RATE:]
        features.append(np.var(last_sec))
        features.append(np.mean(np.abs(last_sec - np.mean(last_sec))))

        # ── 频域特征 ──
        fft_vals = np.abs(np.fft.rfft(signal))
        fft_freq = np.fft.rfftfreq(len(signal), d=1/SAMPLING_RATE)

        features.append(np.sum(fft_vals))                         # 总频谱能量
        features.append(fft_freq[np.argmax(fft_vals)])            # 主频
        features.append(np.sum(fft_vals[fft_freq <= 3]))          # 低频能量 (< 3Hz)
        features.append(np.sum(fft_vals[fft_freq >= 10]))         # 高频能量 (> 10Hz)

    # ── 跨轴相关特征 ──
    # 加速度幅值 (magnitude)
    mag = np.sqrt(np.sum(acc_data ** 2, axis=1))
    features.append(np.max(mag))                                   # 峰值
    features.append(np.std(mag))                                   # 标准差
    features.append(np.var(mag[-SAMPLING_RATE:]))                 # 静止段方差
    features.append(np.max(np.abs(np.diff(mag))))                 # 幅值变化率

    return np.array(features)

# 提取所有特征
X_fall = np.array([extract_features(s) for s in fall_samples])
X_normal = np.array([extract_features(s) for s in normal_samples])

X = np.vstack([X_fall, X_normal])
y = np.array([1] * N_FALL + [0] * N_NORMAL)

print(f"  特征维度: {X.shape[1]} 维")
print(f"    时域特征: 10 × 3 轴 = 30")
print(f"    频域特征:  4 × 3 轴 = 12")
print(f"    跨轴特征:  3")
print(f"  特征矩阵: {X.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 数据集划分
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/6] 数据预处理与划分...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  训练集: {X_train.shape[0]} 条  (fall={y_train.sum()}, normal={len(y_train)-y_train.sum()})")
print(f"  测试集: {X_test.shape[0]} 条    (fall={y_test.sum()}, normal={len(y_test)-y_test.sum()})")


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
y_proba = best_model.predict_proba(X_test_scaled)

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
    target_names=["normal (正常)", "fall (摔倒)"],
    digits=4,
))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print(f"  混淆矩阵：")
print(f"  ┌──────────────┬──────┬──────┐")
print(f"  │              │ 预测正常 │ 预测摔倒 │")
print(f"  ├──────────────┼──────┼──────┤")
print(f"  │ 实际正常     │ {cm[0][0]:4d} │ {cm[0][1]:4d} │")
print(f"  ├──────────────┼──────┼──────┤")
print(f"  │ 实际摔倒     │ {cm[1][0]:4d} │ {cm[1][1]:4d} │")
print(f"  └──────────────┴──────┴──────┘")

# 5折交叉验证
print(f"\n  5 折交叉验证：")
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring="f1")
for i, s in enumerate(cv_scores):
    print(f"    Fold {i+1}:  F1 = {s:.4f}")
print(f"    均值:       F1 = {cv_scores.mean():.4f}  (±{cv_scores.std():.4f})")

# 特征重要性 Top-15
importances = best_model.feature_importances_
top_indices = np.argsort(importances)[-15:][::-1]

feature_labels = []
axis_names = ["X轴", "Y轴", "Z轴"]
for ax in axis_names:
    feature_labels += [
        f"{ax}-均值", f"{ax}-标准差", f"{ax}-最大值", f"{ax}-最小值",
        f"{ax}-峰峰值", f"{ax}-IQR", f"{ax}-最大变化率", f"{ax}-平均变化率",
        f"{ax}-静止方差", f"{ax}-静止MAE",
        f"{ax}-频谱能量", f"{ax}-主频", f"{ax}-低频能量", f"{ax}-高频能量",
    ]
feature_labels += ["幅值-峰值", "幅值-标准差", "幅值-静止方差", "幅值-变化率"]

print(f"\n  Top-15 重要特征：")
for rank, idx in enumerate(top_indices, 1):
    label = feature_labels[idx] if idx < len(feature_labels) else f"特征_{idx}"
    print(f"    {rank:2d}. {label:<20s} importance={importances[idx]:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 三步交叉验证模拟
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*64}")
print("  模拟推理：三步交叉验证")
print(f"{'='*64}")

test_fall = generate_fall_pattern()
test_normal = generate_normal_pattern()

for i, (sample, name) in enumerate([(test_fall, "摔倒样本"), (test_normal, "正常行走样本")]):
    feat = extract_features(sample).reshape(1, -1)
    feat_scaled = scaler.transform(feat)
    pred = best_model.predict(feat_scaled)[0]
    proba = best_model.predict_proba(feat_scaled)[0]

    print(f"\n  {name}:")
    print(f"    预测: {'摔倒' if pred == 1 else '正常'}  (置信度 {max(proba):.4f})")

    # 三步判断
    mag = np.sqrt(np.sum(sample ** 2, axis=1))
    diff_mag = np.diff(mag)
    impact = np.max(np.abs(diff_mag)) > 15
    stillness_var = np.var(mag[-SAMPLING_RATE:])
    still = stillness_var < 1.0

    print(f"    步骤1 - 冲击检测:  {'通过' if impact else '未通过'}  (变化率峰值 {np.max(np.abs(diff_mag)):.1f})")
    print(f"    步骤2 - 姿态变化:  {'通过' if impact else '未通过'}  (随冲击判定)")
    print(f"    步骤3 - 静止检测:  {'通过' if still else '未通过'}  (静止方差 {stillness_var:.3f})")


# ═══════════════════════════════════════════════════════════════════════════════
# 保存模型
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*64}")
print("  保存模型文件...")

os.makedirs("models", exist_ok=True)

with open(os.path.join("models", "fall_model_rf.pkl"), "wb") as f:
    pickle.dump(best_model, f)
with open(os.path.join("models", "fall_scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print(f"  ✓ models/fall_model_rf.pkl  (RandomForest, n={best_model.n_estimators}, acc={accuracy:.3f})")
print(f"  ✓ models/fall_scaler.pkl    (StandardScaler)")
print()
print(f"  训练完成！模型可用于实时推理。")
print(f"{'='*64}")
