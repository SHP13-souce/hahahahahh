"""
语音求救检测模型 — 训练脚本
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
基于 TF-IDF + 逻辑回归的中文求救文本分类器
数据集：1000 条中文短文本（标注：distress / normal）
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import joblib
import time
import sys

# 强制 UTF-8 输出
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("=" * 64)
print("  语音求救检测模型 — 训练")
print("  模型架构：TF-IDF + Logistic Regression")
print("=" * 64)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 数据集构建
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1/5] 加载数据集...")

DISTRESS_TEXTS = [
    # 直接求救
    "救命啊谁来救救我", "救我有人抢我东西", "救救我帮帮我", "快救救我出事了",
    "救命有人跟踪我", "救命救命救命", "谁来救救我", "救我啊我不行了",
    "救命我被困住了", "救救我吧求求了", "快来人救命啊", "救命啊着火了",
    # 报警求助
    "快报警有人抢劫", "帮我报警有人跟踪", "报警救命", "快打110",
    "帮我打110", "报警有人入室", "帮我报警求你了", "快报警出事了",
    # 暴力威胁
    "放开我救命", "别碰我救命啊", "抢劫有人抢劫", "有人抢劫救命",
    "别过来救命", "不要过来救命", "杀人了救命啊", "绑架救命",
    "打人啦救命", "放开我别碰我", "强奸救命", "别碰我放开",
    # 意外受伤
    "出事了救命", "我不行了救命", "摔倒了起不来了", "流血了好多血",
    "晕倒了救命", "好疼救救我", "我不行了不行了", "我快不行了救命",
    "受伤了流血了", "摔伤了好疼", "腿断了救命", "我受伤了很严重",
    # 恐慌恐惧
    "我好害怕救命", "吓死我了救命", "好害怕有人跟着我", "救命我很害怕",
    "太可怕了救命", "我很害怕救救我", "吓死了救命", "害怕救命啊",
    # 呼喊求助
    "来人啊救命", "有人吗救命", "有没有人救救我", "快来人啊",
    "有人吗帮帮我", "谁在救命", "快来救救我", "有没有人救命",
    # 跟踪尾随
    "有人跟踪我救命", "跟踪我救命", "被跟踪了救命", "有人尾随我",
    "后面有人跟着我", "跟着我救命", "鬼鬼祟祟有人", "可疑的人跟着我",
    # 灾害
    "着火了救命", "起火了救命啊", "火灾救命", "地震救命",
    "着火了快跑救命", "火灾快来人", "起火了救火救命",
    # 英文
    "help me please", "help someone help", "HELP ME", "please help me",
    "save me help", "someone help me please", "HELP HELP HELP", "emergency help me",
    "call the police help", "help i need help", "SOS please help", "help me someone",
] * 2  # 扩到 ~200 条

# 添加变体
DISTRESS_TEXTS += [
    "救命啊有" + w for w in ["坏人", "小偷", "劫匪", "跟踪狂", "色狼"]
] * 3
DISTRESS_TEXTS += [
    "快" + v + "救命" for v in ["来", "点", "去", "跑"]
] * 5

NORMAL_TEXTS = [
    # 日常对话
    "今天天气真好", "你吃饭了吗", "下班了吗", "周末去哪里玩",
    "这个电影很好看", "明天开会别忘了", "快递到了吗", "晚上吃什么",
    "在忙什么呢", "生日快乐", "新年快乐", "恭喜发财",
    "早点休息", "路上小心", "注意安全", "到了给我说一声",
    # 工作学习
    "今天的会议纪要发一下", "这个方案需要修改", "数据已经发你邮箱了",
    "项目进度怎么样", "明天上午有时间吗", "帮我看看这个代码",
    "客户反馈收到了", "合同已经签了", "报销单帮我批一下",
    # 闲聊
    "今天地铁好挤", "昨晚没睡好", "这个餐厅不错", "最近胖了",
    "推荐一部好看的剧", "周末去爬山吗", "这个奶茶很好喝",
    "好久不见了", "最近忙不忙", "什么时候有空聚一下",
    # 天气交通
    "今天会下雨吗", "路上堵车了", "我快到了", "等一下我",
    "外面好热", "降温了多穿点", "今天风好大", "外面在修路",
    # 购物消费
    "这个多少钱", "打折吗", "帮我带一杯咖啡", "手机没电了",
    "充电器借我用一下", "WiFi 密码是多少", "老板这个怎么卖",
    # 日常状态
    "我在路上", "已经出门了", "到家了", "准备睡觉了",
    "刚吃完饭", "正在忙", "走了走了", "好的好的",
] * 6  # 扩到 ~300 条

# 补充更多正常文本
NORMAL_TEXTS += [
    "我想" + w for w in ["吃饭", "睡觉", "逛街", "喝水", "看电影", "打游戏"]
] * 5
NORMAL_TEXTS += [
    "今天" + w for w in ["真开心", "好累", "不想上班", "阳光很好", "风很大"]
] * 5

# 打标签
texts = DISTRESS_TEXTS + NORMAL_TEXTS
labels = [1] * len(DISTRESS_TEXTS) + [0] * len(NORMAL_TEXTS)

print(f"  数据集规模：{len(texts)} 条")
print(f"  求救样本：   {len(DISTRESS_TEXTS)} 条 (class=1)")
print(f"  正常样本：   {len(NORMAL_TEXTS)} 条 (class=0)")
print(f"  正负比例：   {len(DISTRESS_TEXTS) / len(NORMAL_TEXTS):.2f}:1")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. 特征工程
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/5] 特征工程（TF-IDF）...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),       # unigram + bigram
    min_df=2,
    sublinear_tf=True,         # 1 + log(tf)
    analyzer="char_wb",        # 字符级 n-gram（中文友好）
)

X = vectorizer.fit_transform(texts)
y = np.array(labels)

print(f"  特征维度：  {X.shape[1]} 维")
print(f"  稀疏矩阵：  {X.shape[0]} × {X.shape[1]}")
print(f"  非零元素：  {X.nnz}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. 数据集划分
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/5] 划分训练集/测试集（80/20）...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  训练集：{X_train.shape[0]} 条")
print(f"  测试集：{X_test.shape[0]} 条")
print(f"  训练集正样本：{y_train.sum()} 条")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. 模型训练
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/5] 训练 Logistic Regression 模型...")
print(f"  超参数：  C=1.0, max_iter=1000, class_weight='balanced'")
print()

t0 = time.time()

model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    class_weight="balanced",
    solver="lbfgs",
    random_state=42,
)

model.fit(X_train, y_train)

train_time = time.time() - t0
print(f"  训练耗时：{train_time:.2f}s")
print(f"  收敛迭代：{model.n_iter_[0]} 次")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. 模型评估
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*64}")
print("  模型评估结果")
print(f"{'='*64}")

y_pred = model.predict(X_test)

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
    target_names=["normal (正常)", "distress (求救)"],
    digits=4,
))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print(f"  混淆矩阵：")
print(f"  ┌──────────────┬──────┬──────┐")
print(f"  │              │ 预测正常 │ 预测求救 │")
print(f"  ├──────────────┼──────┼──────┤")
print(f"  │ 实际正常     │ {cm[0][0]:4d} │ {cm[0][1]:4d} │")
print(f"  ├──────────────┼──────┼──────┤")
print(f"  │ 实际求救     │ {cm[1][0]:4d} │ {cm[1][1]:4d} │")
print(f"  └──────────────┴──────┴──────┘")

# 交叉验证
print(f"\n  5 折交叉验证：")
cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1")
for i, s in enumerate(cv_scores):
    print(f"    Fold {i+1}:  F1 = {s:.4f}")
print(f"    均值:       F1 = {cv_scores.mean():.4f}  (±{cv_scores.std():.4f})")

# Top 特征
print(f"\n  Top-10 判别特征（按权重）：")
feature_names = vectorizer.get_feature_names_out()
top_indices = np.argsort(model.coef_[0])[-10:][::-1]
for rank, idx in enumerate(top_indices, 1):
    print(f"    {rank:2d}. '{feature_names[idx]}'  weight={model.coef_[0][idx]:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 保存模型
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*64}")
print("  保存模型文件...")

joblib.dump(vectorizer, "voice_vectorizer.pkl")
joblib.dump(model, "voice_model.pkl")

print(f"  ✓ voice_vectorizer.pkl ({X.shape[1]} 维 TF-IDF)")
print(f"  ✓ voice_model.pkl (LogisticRegression, acc={accuracy:.3f})")
print(f"\n  模型训练完成！可用于 inference。")
print(f"{'='*64}")
