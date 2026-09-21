#!/usr/bin/env python3
"""
ROS2 语音转发节点 v3.0
- 轮询后端 /mic/status → 检测到前端手动唤醒 → 发布 ROS 话题触发 M2 硬件
- 订阅 awake_flag → 检测到硬件唤醒（喊"小微小微"）→ 通知后端，前端显示聆听
- 订阅 voice_words → 收到 ASR 文字 → 转发给 FastAPI 后端

订阅话题: voice_words, awake_flag
发布话题: wheeltec_mic/wakeup_trigger
后端地址: http://127.0.0.1:8000
"""

import threading
import time

import rclpy
import requests
from rclpy.node import Node
from std_msgs.msg import Int8, String

API_URL = "http://127.0.0.1:8000"

# 端到端语音预算：与 voice_transfer_node.py / voice_agent_adapter 共用同一
# 语义（业务 60s + 清理 grace + 余量）。调用方超时必须大于适配器预算。
try:
    from voice_agent_adapter import VOICE_TURN_TIMEOUT_S as VOICE_CHAT_TIMEOUT_S
except ImportError:  # 仅拷贝节点脚本部署时的回退值，必须与适配器保持同步
    VOICE_CHAT_TIMEOUT_S = 75.0
WAKEUP_TOPIC = "wheeltec_mic/wakeup_trigger"
POLL_INTERVAL = 0.5  # 轮询间隔（秒）


class VoiceTransferNode(Node):
    """ROS2 语音转发节点：桥接前端/后端 与 wheeltec_mic_aiui"""

    def __init__(self):
        super().__init__("voice_to_llm_node")

        self._last_triggered = False
        self._hw_woken = False

        # 创建 M2 唤醒指令发布者
        self._wakeup_pub = self.create_publisher(String, WAKEUP_TOPIC, 10)

        # 订阅 ASR 识别结果
        self._voice_sub = self.create_subscription(
            String, "voice_words", self.voice_callback, 10
        )

        # 订阅硬件唤醒标志（喊"小微小微"时 M2 触发）
        self._awake_sub = self.create_subscription(
            Int8, "awake_flag", self.awake_flag_callback, 10
        )

        # 启动后台轮询线程（检测前端手动唤醒）
        self._poll_thread = threading.Thread(target=self.poll_mic_status, daemon=True)
        self._poll_thread.start()

        self.get_logger().info("语音转发节点 v3.0 (ROS2) 已启动")
        self.get_logger().info("  - 订阅: voice_words, awake_flag")
        self.get_logger().info(f"  - 发布: {WAKEUP_TOPIC}")
        self.get_logger().info(f"  - 后端: {API_URL}")

    def poll_mic_status(self):
        """后台线程：轮询后端 /mic/status，检测到前端手动唤醒 → 发布 ROS 话题触发 M2"""
        while rclpy.ok():
            try:
                resp = requests.get(f"{API_URL}/mic/status", timeout=2)
                data = resp.json()
                triggered = data.get("triggered", False)

                # 上升沿：刚触发
                if triggered and not self._last_triggered:
                    self.get_logger().info(
                        f"[MIC] 检测到前端唤醒请求 → 发布 {WAKEUP_TOPIC}"
                    )
                    msg = String()
                    msg.data = "wakeup"
                    self._wakeup_pub.publish(msg)
                    self.get_logger().info("[MIC] 已发送唤醒指令")

                self._last_triggered = triggered

            except requests.exceptions.ConnectionError:
                self.get_logger().warn(
                    "[MIC] 无法连接后端，请确认 qa_server.py 已启动",
                    throttle_duration_sec=30.0,
                )
            except Exception as e:  # noqa: BLE001 — 常驻轮询线程必须存活
                self.get_logger().warn(
                    f"[MIC] 轮询异常: {e}",
                    throttle_duration_sec=30.0,
                )

            time.sleep(POLL_INTERVAL)

    def awake_flag_callback(self, msg: Int8):
        """检测到硬件唤醒（喊"小微小微"）→ 通知后端，让前端显示聆听状态"""
        if msg.data == 1 and not self._hw_woken:
            self._hw_woken = True
            self.get_logger().info(
                "[MIC] 检测到硬件语音唤醒（小微小微）→ 通知前端显示聆听"
            )
            try:
                requests.post(f"{API_URL}/mic/hw_wakeup", timeout=2)
            except Exception:  # noqa: BLE001, S110 — 通知前端失败无需惊动用户
                pass

    def voice_callback(self, msg: String):
        """收到 M2 的 ASR 识别结果 → 通知后端 + 转发给问答后端"""
        text = msg.data.strip()
        if not text:
            return

        self.get_logger().info(f"麦克风识别: {text}")

        # 回传给后端（供前端轮询获取 ASR 文字 + SSE 推送 mic_status 事件）
        try:
            requests.post(f"{API_URL}/mic/notify_asr", json={"text": text}, timeout=3)
        except Exception:  # noqa: BLE001, S110 — 回传失败不影响问答主链
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
            self.get_logger().info(f"回复 [{label}]: {answer}")
            print(f"\n问题: {text}\n来源: {label}\n回答: {answer}\n")
        except requests.exceptions.ConnectionError:
            self.get_logger().error(
                f"无法连接后端 ({API_URL})，请确认 qa_server.py 已启动"
            )
        except Exception as err:  # noqa: BLE001 — 语音回调不得因单次请求异常中断
            self.get_logger().error(f"请求失败: {err}")

        # 本轮结束，重置硬件唤醒标记
        self._hw_woken = False


def main():
    rclpy.init()
    node = VoiceTransferNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
