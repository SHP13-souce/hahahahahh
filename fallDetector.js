/**
 * 摔倒检测模块 — 路演版
 * -----------------------
 * 只产出数据，不调用后端。调用方拿到 fallResult 后自行合并进
 * riskPayload.sensor → POST /api/risk/analyze
 *
 * 用法：
 *   import { startFallDetection, triggerFallForDemo } from './fallDetector.js'
 *
 *   // 真实传感器
 *   startFallDetection((result) => {
 *     riskPayload.sensor = { ...riskPayload.sensor, ...result }
 *     await fetch('/api/risk/analyze', { body: JSON.stringify(riskPayload) })
 *   })
 *
 *   // 路演兜底
 *   triggerFallForDemo((result) => { ... })
 */

let lastAcc = null
let impactDetected = false
let stillSeconds = 0
let stillTimer = null

const IMPACT_THRESHOLD = 2.5  // 加速度突变 > 2.5g → 冲击
const STILL_DURATION = 5      // 持续静止 ≥ 5 秒 → 确认摔倒

// ── 公开 API ──────────────────────────────────────────────────────────────────

/**
 * 开启真实传感器摔倒检测
 * 冲击 → 计时 5 秒静止 → callback(fallResult)
 */
export function startFallDetection(callback) {
  reset()

  uni.startAccelerometer({
    interval: 'normal',
    success: () => console.log('[FallDetector] 加速度计已开启'),
    fail: (err) => console.warn('[FallDetector] 开启失败', err),
  })

  uni.onAccelerometerChange((res) => {
    const { x, y, z } = res
    const totalAcc = Math.sqrt(x * x + y * y + z * z)

    let delta = 0
    if (lastAcc !== null) {
      delta = Math.abs(totalAcc - lastAcc)
    }
    lastAcc = totalAcc

    if (!impactDetected && delta > IMPACT_THRESHOLD) {
      impactDetected = true
      startStillTimer(callback)
    }
  })
}

/**
 * 路演兜底 — 跳过传感器，直接返回摔倒结果
 */
export function triggerFallForDemo(callback) {
  impactDetected = true
  stillSeconds = STILL_DURATION
  callback(buildResult(true, '检测到疑似摔倒（演示模式）'))
}

/**
 * 停止检测，清理资源
 */
export function stopFallDetection() {
  clearStillTimer()
  reset()

  try { uni.stopAccelerometer({ fail: () => {} }) } catch (e) { /* 静默 */ }

  console.log('[FallDetector] 已停止')
}

// ── 内部 ──────────────────────────────────────────────────────────────────────

function startStillTimer(callback) {
  clearStillTimer()
  stillSeconds = 0

  stillTimer = setInterval(() => {
    stillSeconds += 1

    if (stillSeconds >= STILL_DURATION) {
      clearStillTimer()

      try { uni.stopAccelerometer({ fail: () => {} }) } catch (e) { /* 静默 */ }

      callback(buildResult(true, '检测到疑似摔倒'))
    }
  }, 1000)
}

function clearStillTimer() {
  if (stillTimer) {
    clearInterval(stillTimer)
    stillTimer = null
  }
}

function reset() {
  lastAcc = null
  impactDetected = false
  stillSeconds = 0
}

function buildResult(detected, message) {
  return {
    fallDetected: detected,
    shakeLevel: detected ? 'high' : 'normal',
    stillSeconds,
    impactDetected,
    postureChanged: detected,
    message,
  }
}
