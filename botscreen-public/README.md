# botscreen

天津市眼科医院视光中心的互动展示与问答桌面应用。项目使用 Electron 承载 Vue 3 页面，提供视光科普内容展示、人员与特色技术介绍，以及面向儿童的语音/点击问答界面。

问答服务是一个独立的 FastAPI 进程：问题优先从本地知识库匹配，未命中时才请求 DeepSeek。Electron 前端通过 HTTP 和 Server-Sent Events（SSE）接收回答与麦克风状态。

## 功能概览

- 首页导航：进入互动问答、视光科普、特色技术和人员介绍。
- 互动问答：点击推荐问题或使用麦克风唤醒；展示问答结果，并支持浏览器中文语音播报。
- 本地知识库：问答服务启动时读取 DOCX；DOCX 不存在时使用内置的儿童视光科普条目。
- 大模型兜底：本地知识库未命中时调用 DeepSeek，并在返回结果中标记来源。
- 视光科普：从 YAML 加载视频卡片，支持标签筛选和视频链接预览。
- 内容驱动页面：人员、特色技术、图片、视频和 Markdown 详情均可放在外部资源目录中，不必重新编译应用。
- 机器人语音桥接：提供 ROS1 和 ROS2 两套语音转发节点及一键启动脚本。
- kiosk 模式：生产构建默认全屏、无边框并保持窗口置顶，适合机器人或公共屏幕设备。

## 技术栈

| 层级 | 技术 |
| --- | --- |
| 桌面容器 | Electron 40、electron-vite |
| 前端 | Vue 3、TypeScript、Vue Router |
| 样式与动效 | Tailwind CSS、Motion、GSAP、OGL、Lucide |
| 内容解析 | YAML、Markdown-it、脚注、目录、Mermaid |
| 问答后端 | Python、FastAPI、Uvicorn、jieba、Requests |
| 语音/机器人 | ROS1 或 ROS2、Wheeltec 麦克风节点（可选） |
| 打包 | electron-builder（Windows、macOS、Linux） |

## 运行要求

### Electron 前端

- Node.js 22 或 24 LTS，并确保 npm 版本兼容。
- 能够安装 Electron 依赖的桌面环境。
- 如果要显示外部资源，需要准备一个资源目录并通过 RCPATH 指向它。

### 问答后端

- Python 3.10 或更新版本（源码使用现代类型标注）。
- Python 依赖见 server/requirements.txt。
- 可选的 DeepSeek API Key；仅当问题未命中本地知识库时才会使用。

### 机器人语音（可选）

- ROS1 或 ROS2 工作空间。
- Wheeltec 麦克风/讯飞 AIUI 节点和对应的 M2 音频设备。
- ROS 语音话题与脚本中的话题名称保持一致。

## 快速开始

### 1. 安装依赖

~~~bash
git clone https://github.com/Snow7-bot/botscreen-public.git
cd botscreen-public
npm ci
~~~

如果没有 package-lock.json 或需要更新依赖，也可以使用 npm install。首次安装会执行 electron-builder 的 postinstall 步骤。

### 2. 启动问答后端

在单独的终端中执行：

~~~bash
cd server
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python qa_server.py
~~~

Windows PowerShell 的虚拟环境激活命令为：

~~~powershell
cd server
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python qa_server.py
~~~

服务默认监听 0.0.0.0:8000。前端当前将问答服务地址固定为 http://127.0.0.1:8000，因此开发时请保持 8000 端口，或同步修改 src/renderer/src/pages/qa/index.vue。

### 3. 启动 Electron 开发环境

在仓库根目录执行：

~~~bash
DEBUG=1 npm run dev
~~~

DEBUG=1 会关闭全屏 kiosk 行为，便于开发调试。Windows PowerShell 可先执行 $env:DEBUG='1'，再运行 npm run dev。只运行前端时可以先不启动问答后端，但 /qa 页面中的实时问答、推荐问题和麦克风状态会显示连接失败或使用前端内置的推荐问题。

## 后端配置

qa_server.py 支持以下环境变量：

| 变量 | 是否必需 | 作用 |
| --- | --- | --- |
| KB_DOCX_PATH | 否 | 问答 DOCX 路径；未设置时使用代码中的 Windows 默认路径。文件不存在时使用内置知识库。 |
| DEEPSEEK_API_KEY | 否 | DeepSeek 认证 Key。未设置时，命中本地知识库的问题仍可回答，未命中的问题会进入错误兜底。 |

示例（只使用占位符，不要把真实 Key 写入仓库）：

~~~bash
export KB_DOCX_PATH=/opt/botscreen/config/机器人-科普问答.docx
export DEEPSEEK_API_KEY=YOUR_DEEPSEEK_API_KEY
python server/qa_server.py
~~~

问答服务的处理顺序如下：

1. 启动时从 KB_DOCX_PATH 解析段落和问答对。
2. 使用 jieba 分词、关键词和中文字符二元组计算匹配分数。
3. 分数达到阈值时直接返回本地答案，响应中的 source 为 kb。
4. 未命中时请求 DeepSeek，响应中的 source 为 deepseek；请求失败或超时则为 error。
5. 每个 session 最多保留 10 轮历史，30 分钟无活动后清理。

仓库中的 server/knowledge_base.yml 是可供维护者参考的知识内容文件；当前 qa_server.py 的实际加载入口是 DOCX 和内置条目，修改该 YAML 不会自动改变运行时知识库，除非同时扩展加载逻辑。

## 资源目录与 RCPATH

Electron 主进程通过 rc:// 协议读取外部内容。设置 RCPATH 后，应用会将它作为资源根目录；未设置时使用当前工作目录。打包部署时建议始终显式设置 RCPATH。

一个可用的资源目录可以是：

~~~text
<resource-root>/
├── bg.png                         # 首页和问答页背景
├── bg-blank.png                   # 备用背景
├── photo.png                      # 卡片缺省图片（可选）
├── departments/
│   ├── departments.yml            # 特色技术列表
│   └── *.md                       # 技术详情（可选）
├── members/
│   ├── members.yml                # 人员列表
│   ├── *.md                       # 人员详情（可选）
│   └── *.(png|jpg|webp)           # 人员图片
└── science-video/
    ├── index.yml                  # 科普视频卡片列表
    └── *.(png|jpg|webp|mp4)        # 图片或本地视频（可选）
~~~

当前页面读取的固定文件名为：

- departments/departments.yml
- members/members.yml
- science-video/index.yml

### YAML 示例

特色技术 departments/departments.yml：

~~~yaml
- intro: "用一句话介绍技术"
  officer: "负责人"
  role: "职务"
  dept: "技术名称"
  detail: "departments/technology.md"
~~~

人员 members/members.yml：

~~~yaml
- intro: "人员简介"
  name: "姓名"
  title: "职称或岗位"
  tag: ["视光", "儿童"]
  image: "rc://members/example.webp"
  detail: "members/example.md"
~~~

科普视频 science-video/index.yml：

~~~yaml
- title: "视频标题"
  badge: "科普"
  description: "视频简介"
  tags: ["近视防控"]
  date: "2026-01-01"
  image: "rc://science-video/example.webp"
  video: "https://example.com/video"
~~~

图片或本地视频可以使用 rc:// 资源地址；当前版本只允许本地 rc:// 视频通过 `<video>` 播放，外部 URL 暂不启用。

仓库中的 public 符号链接目标是 Windows 风格的 D:/config。它在 macOS/Linux 上通常不可用，请使用 RCPATH 指向实际资源目录，或在部署环境中重新创建对应链接。

## API 文档与接口

完整接口定义见 [OpenAPI 3.0.3](server/openapi.yaml)。可直接打开的 Swagger 页面见 [server/swagger.html/swagger.html](server/swagger.html/swagger.html)，根目录也提供了一个独立的 [swagger.html](swagger.html)。

服务端路由如下：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | /chat | 提交问题；返回 user_question、robot_answer 和 source。请求体至少包含 question，可选 session_id。 |
| GET | /suggestions | 返回知识库前 6 个推荐问题。 |
| POST | /mic/wakeup | 前端手动唤醒麦克风。 |
| POST | /mic/hw_wakeup | ROS/ROS2 硬件语音唤醒回调。 |
| POST | /mic/stop | 停止当前麦克风会话并清除状态。 |
| GET | /mic/status | 查询唤醒状态、已用时和最近一次 ASR 文本。 |
| POST | /mic/notify_asr | ROS/ROS2 节点回传 ASR 文本，请求体为 { "text": "..." }。 |
| GET | /sse | 订阅问答答案和 mic_status 事件的 SSE 长连接。 |
| GET | /health | 返回服务版本、知识库条数和匹配阈值。 |

快速检查：

~~~bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/suggestions
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"什么是近视？","session_id":"demo"}'
curl -N http://127.0.0.1:8000/sse
~~~

## ROS/ROS2 语音桥接

两个脚本都会启动 FastAPI 后端、Wheeltec 麦克风节点、语音转发节点和 Electron 前端：

~~~bash
cd server
chmod +x start_robot.sh start_robot_ros2.sh
./start_robot.sh          # ROS1
./start_robot_ros2.sh     # ROS2
~~~

启动前请按设备修改脚本中的工作空间路径：

- ROS1：脚本默认进入 ~/wheeltec_ros，并执行 wheeltec_mic_aiui 的 aiui_chat.launch。
- ROS2：脚本默认加载 ~/ros2_ws/install/setup.bash，并执行 aiui_chat.launch.py。

ROS1 节点订阅 /voice_words 和 /awake_flag，发布 /wheeltec_mic/wakeup_trigger。ROS2 节点订阅 voice_words 和 awake_flag，发布 wheeltec_mic/wakeup_trigger。两者都会把 ASR 文本发送到 /mic/notify_asr，再调用 /chat。

没有 ROS 或实体麦克风时，/qa 页面上的按钮仍可测试后端唤醒状态，但不会自行完成语音识别；可以通过 HTTP 直接调用 /mic/notify_asr 模拟 ASR 回调。

## 构建与打包

常用命令：

~~~bash
npm run typecheck       # Node、preload、renderer 类型检查
npm run lint            # ESLint
npm run format          # Prettier 格式化（会修改文件）
npm run build           # 类型检查 + electron-vite 构建
npm run build:unpack    # 构建未打包目录
npm run build:win       # Windows 安装包
npm run build:mac       # macOS 安装包
npm run build:linux     # Linux AppImage、snap、deb
npm run start           # 预览已构建的应用
~~~

构建产物默认写入 dist/，未打包的 Linux 可执行文件通常位于 dist/linux-unpacked/botscreen。跨平台打包应在对应目标系统上执行，并准备好该平台的 Electron 构建工具链。

### Linux Wayland kiosk 示例

如果设备使用 cage/Wayland，可以在启动前设置：

~~~bash
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export LIBSEAT_BACKEND=logind
export ELECTRON_OZONE_PLATFORM_HINT=wayland
export RCPATH=/opt/botscreen/config

cage -s -- \
  /opt/botscreen/dist/linux-unpacked/botscreen \
  --ozone-platform=wayland \
  --enable-features=UseOzonePlatform
~~~

这些变量只适用于相应的 Linux 图形环境；普通桌面开发直接使用 npm run dev 即可。

## 代码结构

~~~text
.
├── src/
│   ├── main/                    # Electron 主进程、rc:// 协议和 IPC
│   ├── preload/                # contextBridge 暴露的安全 API
│   └── renderer/src/
│       ├── pages/               # 基于文件的 Vue Router 页面
│       ├── components/          # 页面组件和 UI 组件
│       └── assets/              # CSS 与 Markdown 样式
├── server/
│   ├── qa_server.py             # FastAPI 问答服务
│   ├── voice_transfer_node.py   # ROS1 语音转发
│   ├── voice_transfer_node_ros2.py
│   ├── start_robot*.sh          # 机器人端一键启动脚本
│   ├── openapi.yaml             # API 规范
│   └── swagger.html/            # Swagger UI 页面
├── build/                       # Electron 图标与 macOS entitlements
├── resources/                   # 打包资源（当前包含应用图标）
├── electron-builder.yml         # 安装包配置
├── electron.vite.config.ts      # Vite 与 Vue 路径配置
└── package.json                 # 脚本与依赖
~~~

主进程和 preload 只通过已注册的 IPC 通道访问 YAML/Markdown；读取路径会限制在资源根目录内，并检查扩展名。渲染进程页面使用 @renderer 别名和文件式路由。

## 开发建议

1. 修改页面或 TypeScript 后运行 npm run typecheck 和 npm run lint。
2. 修改内容资源后重启 qa_server.py 或重新加载 Electron 页面。
3. 新增 Markdown 时将文件放在 RCPATH 内，并在 YAML 中使用相对路径。
4. 新增本地图片或视频时优先使用 rc:// 路径，避免把大体积运行时资源提交到源码仓库。
5. 提交前运行 git diff --check，确认没有空白字符错误。

项目当前没有配置自动化单元测试脚本；运行时联调至少应检查 /health、/suggestions、/chat 和 /sse。

## 常见问题

### 页面提示 Yaml file not found

确认 RCPATH 指向资源根目录，并检查 departments/departments.yml、members/members.yml 和 science-video/index.yml 是否存在。未设置 RCPATH 时，应用会在当前工作目录查找这些文件。

### /qa 页面无法连接后端

确认已在 server 目录启动 python qa_server.py，并检查 8000 端口是否被其他程序占用。前端 API 地址当前写死为 127.0.0.1:8000。

### 未命中问题返回 error

这是预期的兜底行为。先确认问题是否属于本地知识库；如果需要 DeepSeek 兜底，请在启动后端的同一终端中设置 DEEPSEEK_API_KEY。不要把 Key 写入 .ts、.py、YAML、README 或提交历史。

### 语音唤醒没有反应

先用 curl 验证 /mic/wakeup、/mic/status 和 /mic/notify_asr，再检查 ROS/ROS2 工作空间、Wheeltec 节点、话题名称和设备权限。没有硬件时只能验证 HTTP 链路。

### 生产应用无法像普通窗口一样关闭

非 DEBUG 环境会启用全屏 kiosk 行为，这是设备部署设计；即使开发模式未设置 DEBUG，也会使用该行为。开发调试请设置 DEBUG=1；不要在生产环境通过修改源码暴露或绕过退出控制。

### 外部视频无法显示

当前科普视频通过本地 `<video>` 播放 rc:// 视频文件，不再使用 iframe 打开外部 URL。

## 安全与隐私

- 这是公开仓库，禁止提交 API Key、SSH 私钥、Cookie、患者信息或未授权的人员图片/视频。
- DeepSeek 请求可能包含用户问题；接入真实场景前请完成数据脱敏、告知和合规评估。
- qa_server.py 监听 0.0.0.0，且当前 CORS 允许任意来源；仅在受控网络中使用。对外部署前应限制监听地址、CORS、访问控制和防火墙规则。
- 本地知识库和生成式回答都不能替代医生诊断。对外展示前请由专业人员审核内容，并保留 AI 生成结果的提示语。
- 仓库当前未提供 LICENSE 文件。二次分发或商业使用前，请先补充并确认合适的开源许可和第三方素材授权。

## 贡献

欢迎提交 Issue 或 Pull Request。建议在 PR 中说明：

- 修改的页面、接口或资源目录；
- 是否需要 ROS/ROS2、外部资源或环境变量；
- 已运行的 typecheck、lint、构建和接口检查；
- 是否涉及医疗内容、个人信息或第三方素材。

## 维护信息

- 应用包名：botscreen
- 当前版本：1.0.0
- 默认后端地址：http://127.0.0.1:8000
- 默认分支：main

## 语音回答后端：legacy / agent 双模式（#58 语音快速切片）

机器人语音链路（M260C + 讯飞 AIUI → ROS → `/mic/*` → `/chat` → `/sse` → 页面 TTS 播报）保持不变，仅 `/chat` 的**回答引擎**可通过环境变量切换：

| 变量 | 说明 |
|---|---|
| `GCMW_VOICE_ANSWER_BACKEND` | `legacy`（默认，KB → DeepSeek，行为不变）或 `agent`（走新 Agent：Manager → RAG → Verifier） |
| `GCMW_VOICE_AGENT_API_BASE` | agent 模式的后端地址，当前固定为 `http://127.0.0.1:8001/api/v1` |
| `GCMW_VOICE_AGENT_CREDENTIAL` | Agent 设备凭据（只进入 `Authorization` 头，缺失/占位符时拒绝启动） |

双服务启动顺序（agent 模式）：

1. **8001 新 Agent**：`PYTHONPATH=. python3 scripts/demo_showcase.py --serve --port 8001 --cors-for-dev-frontends`
2. **8000 兼容服务**：设好上面三个变量后 `python qa_server.py`

要点：

- agent 模式下任何故障（8001 不可达 / 超时 / 证据不足）都返回**固定安全文案**，**不会回退**到 DeepSeek；
- 未通过 Verifier 核验的医疗内容不会显示或播报（引用三元组门槛）；
- **超时模型（第四轮整改）**：整轮问答有 60 秒业务 deadline，作为客户端等待的硬上界，经 `asyncio.timeout_at` 取消域执行——到点时在途请求协程被**真实取消**并关闭连接（不是"停止等待"）。分布式 POST 超时属于**结果未知**（服务端可能已提交，客户端取消不能证明"没有写入"）：Session POST 携带每轮 `idempotency_key`，清理阶段用同一键重放对账，拿到 Session 后按失败路径级联删除；禁止换新键重试（会创建第二个 Session）；
- 清理为有界可取消 best-effort（共享 4s 清理总期限，到点真取消）：**成功终态零 DELETE**——Session/Run/事件保留可审计，由 8001 的**主动 TTL sweeper**（每 5s 扫描，`session_expiry_fault` 固定类别日志）到期回收；失败轮次才删除 Session（级联删 Run 与事件），不留痕；
- 8001 侧 Session 创建支持可选 `idempotency_key`（1–128 字符）：同键同载荷重放返回原 Session（仍 201），同键异载荷返回 409 `E_CONFLICT_IDEMPOTENCY` 零写入，键按 tenant/device 隔离；Session 删除与 TTL 回收同步清理幂等索引；
- 语音回答为「一次性问答」：每个问题一个临时 Agent 会话，当前**不含**多轮语音记忆、流式 ASR/TTS、打断播报、真实硬件验收与生产持久化审计。
