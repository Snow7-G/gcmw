#!/usr/bin/env python3
"""
ROS 语音转发节点 v3.0
- 轮询后端 /mic/status → 检测到前端手动唤醒 → 发布 ROS 话题触发 M2 硬件
- 订阅 /awake_flag → 检测到硬件唤醒（喊"小微小微"）→ 通知后端，前端显示聆听
- 订阅 /voice_words → 收到 ASR 文字 → 转发给 FastAPI 后端

订阅话题: /voice_words, /awake_flag
发布话题: /wheeltec_mic/wakeup_trigger
后端地址: http://127.0.0.1:8000
"""

import threading
import time

import requests
import rospy
from std_msgs.msg import Int8, String

API_URL = "http://127.0.0.1:8000"

# 端到端语音预算：与 voice_agent_adapter 对齐（业务 60s + 清理 grace + 余量）。
# 调用方超时必须大于适配器预算，否则旧请求未结束就允许下一次唤醒。
try:
    from voice_agent_adapter import VOICE_TURN_TIMEOUT_S as VOICE_CHAT_TIMEOUT_S
except ImportError:
    # 仅拷贝节点脚本部署（适配器不在同目录）时的回退值。约束是**不得小于**
    # 适配器公布的调用方超时：小了就是调用方先超时、适配器仍在跑，即第四轮
    # 修过的「迟到副作用 / 结果未知」。由 tests/test_voice_agent_adapter.py 的
    # TestVoiceTurnTimeoutContract 守这个方向，不要只改这个数。
    VOICE_CHAT_TIMEOUT_S = 75.0
WAKEUP_TOPIC = "/wheeltec_mic/wakeup_trigger"
POLL_INTERVAL = 0.5  # 轮询间隔（秒）

_last_triggered = False
_wakeup_pub = None
_hw_woken = False  # 是否已由硬件唤醒（防止重复通知）


def poll_mic_status():
    """后台线程：轮询后端 /mic/status，检测到唤醒标记 → 发布 ROS 话题触发 M2"""
    global _last_triggered

    while not rospy.is_shutdown():
        try:
            resp = requests.get(f"{API_URL}/mic/status", timeout=2)
            data = resp.json()
            triggered = data.get("triggered", False)

            # 上升沿：刚触发
            if triggered and not _last_triggered:
                rospy.loginfo("[MIC] 检测到前端唤醒请求 → 发布 %s", WAKEUP_TOPIC)
                if _wakeup_pub is not None:
                    _wakeup_pub.publish(String(data="wakeup"))
                    rospy.loginfo("[MIC] 已发送唤醒指令")
                else:
                    rospy.logerr("[MIC] Publisher 未初始化")

            _last_triggered = triggered

        except requests.exceptions.ConnectionError:
            rospy.logwarn_throttle(30, "[MIC] 无法连接后端，请确认 qa_server.py 已启动")
        except Exception as e:  # noqa: BLE001 — 常驻轮询线程必须存活，异常只降级为告警
            rospy.logwarn_throttle(30, "[MIC] 轮询异常: %s", e)

        time.sleep(POLL_INTERVAL)


def voice_callback(msg: String) -> None:
    """收到 M2 的 ASR 识别结果 → 通知后端 + 转发给问答后端"""
    global _hw_woken
    text = msg.data.strip()
    if not text:
        return

    rospy.loginfo("麦克风识别: %s", text)

    # 回传给后端（供前端轮询获取 ASR 文字 + SSE 推送 mic_status 事件）
    try:
        requests.post(f"{API_URL}/mic/notify_asr", json={"text": text}, timeout=3)
    except Exception:  # noqa: BLE001, S110 — 回传失败不影响问答主链，静默是设计
        pass

    # 发送给问答后端
    try:
        res = requests.post(
            f"{API_URL}/chat", json={"question": text}, timeout=VOICE_CHAT_TIMEOUT_S
        )
        res.raise_for_status()
        data = res.json()
        answer = data.get("robot_answer", "")
        source = data.get("source", "unknown")
        label = (
            "本地知识库"
            if source == "kb"
            else (
                "DeepSeek"
                if source == "deepseek"
                else ("安全Agent" if source == "agent" else "错误")
            )
        )
        rospy.loginfo("回复 [%s]: %s", label, answer)
        print(f"\n问题: {text}\n来源: {label}\n回答: {answer}\n")
    except requests.exceptions.ConnectionError:
        rospy.logerr("无法连接后端 (%s)，请确认 qa_server.py 已启动", API_URL)
    except Exception as err:  # noqa: BLE001 — 语音回调不得因单次请求异常中断
        rospy.logerr("请求失败: %s", str(err))

    # 本轮结束，重置硬件唤醒标记
    _hw_woken = False


def awake_flag_callback(msg: Int8) -> None:
    """检测到硬件唤醒（喊"小微小微"）→ 通知后端，让前端显示聆听状态"""
    global _hw_woken
    if msg.data == 1 and not _hw_woken:
        _hw_woken = True
        rospy.loginfo("[MIC] 检测到硬件语音唤醒（小微小微）→ 通知前端显示聆听")
        try:
            requests.post(f"{API_URL}/mic/hw_wakeup", timeout=2)
        except Exception:  # noqa: BLE001, S110 — 通知前端失败无需惊动用户
            pass


if __name__ == "__main__":
    rospy.init_node("voice_to_llm_node", anonymous=True)

    # 创建 M2 唤醒指令发布者（只创建一次）
    _wakeup_pub = rospy.Publisher(WAKEUP_TOPIC, String, queue_size=1)
    rospy.sleep(0.3)  # 等订阅者连上

    # 订阅 ASR 识别结果
    rospy.Subscriber("/voice_words", String, voice_callback)

    # 订阅硬件唤醒标志（喊"小微小微"时 M2 触发）
    rospy.Subscriber("/awake_flag", Int8, awake_flag_callback)

    # 启动后台轮询线程（检测前端手动唤醒）
    poll_thread = threading.Thread(target=poll_mic_status, daemon=True)
    poll_thread.start()

    rospy.loginfo("语音转发节点 v3.0 已启动")
    rospy.loginfo("  - 订阅: /voice_words, /awake_flag")
    rospy.loginfo("  - 发布: %s", WAKEUP_TOPIC)
    rospy.loginfo("  - 后端: %s", API_URL)
    rospy.spin()
