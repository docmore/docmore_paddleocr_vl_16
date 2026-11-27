# 验证与记忆文档

> 适用版本：0.1.0 · 最后更新：2026-11-26
>
> 本文记录“为什么长成这样”以及“当时怎么验证的”：需求、实现路径、实测
> 矩阵、踩过的坑、设计理由和已知限制。它不是 API 文档；接口承诺见
> [契约文档](contracts.md)，架构见 [架构文档](architecture.md)。

## 1. 目的

这份文档解决三个维护问题：

1. **回归时知道看哪里**：哪些行为是实测确认过的，哪些只是理论推断；
2. **避免重复踩坑**：每个坑记录症状、根因、修复和回归方式；
3. **保留设计上下文**：有些选择不是“最优解”，而是在当时约束下的取舍。

## 2. 需求与实现

### 2.1 原始需求

以 `docmore_foo` 的统一形式新建 `docmore_paddleocr_vl_16`，全页解析采用
两阶段流水线，两个阶段使用独立模型：

- 第一阶段 PP-DocLayoutV3 检测文本/表格/公式/图表/印章等区域并给出阅读
  顺序；
- 第二阶段把裁剪区域送到 PaddleOCR-VL 1.6 的 OpenAI 兼容服务
  （`api_base: http://127.0.0.1:8113`），按区域类型匹配提示词；
- 最后按阅读顺序组装 Markdown / JSON。

需求给出的提示词集合：

```python
"ocr": "OCR:",
"table": "Table Recognition:",
"formula": "Formula Recognition:",
"chart": "Chart Recognition:",
"spotting": "Spotting:",
"seal": "Seal Recognition:",
```

### 2.2 实现路径

1. 以已经跑通的两阶段实现 `docmore_minicpm_ocr` 为工程底稿（它本身遵循
   `docmore_foo` 的统一形式：`convert()`、`generate()`、CLI/FastAPI、
   docling bridge、页面资产键、URL 取图与 SSRF 防护），整体替换命名空间为
   `docmore.paddleocr_vl_16`。
2. 保留第一阶段：PP-DocLayoutV3 ONNX、25 类 label、NMS/unclip/merge、
   重叠过滤、局部阅读顺序修正、锁内 CPU 回退。
3. 替换第二阶段的任务层：把 MiniCPM-OCR 的
   `Text Recognition:` / `Chart to Table:` 改成 PaddleOCR-VL 1.6 的
   `OCR:` / `Chart Recognition:`，并新增 `Spotting:`。
4. 按实测输出调整后处理：PaddleOCR-VL 1.6 的表格是裸 OTSL（无
   `<otsl>` 包装），图表是 Markdown 管道表或说明文字，都需要不同的兜底。
5. 无版面回退 label 从 `text` 改为 `ocr`；行内公式改用
   `Formula Recognition:`，后处理仍渲染为行内 `$...$`。
6. 对外模型名收敛为六个 `paddleocr-vl-1.6-*` 别名：`doc` 走两阶段，其余
   五个整图单块并强制对应 prompt；`/v1/models`、`/health`、请求校验和
   测试全部按六别名契约实现，非别名返回 404 `model_not_found`。
7. 服务端模型 id、环境变量前缀、端口、包名、CLI 名称、文档和测试全部
   改为本包自己的命名（`PADDLEOCR_VL_16_SERVER_`、默认端口 18910）。
8. 重写 README 与 docs，避免继承上游工程的版本历史与实测结论。

## 3. 验证记录

### 3.1 环境

| 项 | 值 |
| --- | --- |
| OS | Windows 11 |
| Python | 3.12 |
| 包管理 | uv |
| 上游 | 本地 llama.cpp OpenAI 兼容端点，`127.0.0.1:8113` |
| 版面 | PP-DocLayoutV3 ONNX（同级 `docmore_paddleocr_vl/models/PP-DocLayoutV3-onnx`） |

### 3.2 自动化测试

```bash
uv sync --all-extras
uv run python -m pytest -q
# 117 passed
```

默认依赖（没有 `[server]` / `[docling]` / `[layout]` 的 cv2）时：

```text
72 passed, 10 skipped
```

skip 分别来自 Docling 导出、cv2 多边形回归，以及 FastAPI 服务端测试模块
（模块级 `pytest.importorskip`）。发布前必须用 `--all-extras` 跑全量。

测试集覆盖：

| 文件 | 覆盖点 |
| --- | --- |
| `tests/test_layout_pipeline.py` | 区域路由、提示词计数、阅读顺序、图片资产、页号前缀、`asset` 键、页眉开关、印章关闭、强制 prompt（含 spotting）、无版面回退 label、`max_box_w` 回归 |
| `tests/test_mode.py` | 六个模型别名、模式解析、整图单块绕过版面、印章/图表开关覆盖、显式 prompt 优先级 |
| `tests/test_pipeline.py` | `convert()` 端到端、md/html/json/metadata、Docling、目录输入、非法格式/扩展名、旧资产键去重 |
| `tests/test_postprocess.py` | OTSL→HTML（带/不带 wrapper）、HTML 文档包装、管道表兜底、图表说明文本、公式定界符、Markdown 渲染、图片文件名 |
| `tests/test_client.py` | base URL 归一化、退化重复检测 |
| `tests/test_parse_endpoint.py` | `/v1/parse` 请求/响应、图片资产、`asset` 精确匹配、`pages`/bbox/页眉选项、非法选项、图表 caption、`ocr` label、错误路径 |
| `tests/test_server_api.py` | `/v1/models`、`/health`、鉴权、chat completions、部分失败 |
| `tests/test_server_concurrency.py` | `asyncio.to_thread` 生效：并发请求重叠、`/health` 不被拖住 |
| `tests/test_url_sources.py` | URL 下载、重定向、大小/超时、SSRF 私网拒绝、每跳校验、`/v1/parse` URL 集成 |

测试不依赖真实模型：`tests/mock_server.py` 用标准库实现 OpenAI 兼容 mock，
按 prompt 返回文本 / OTSL 表格 / OTSL 图表 / LaTeX / 印章 / spotting；
版面检测在测试中被关闭或桩替换。

### 3.3 构建验证

```bash
uv build
# dist/docmore_paddleocr_vl_16-0.1.0-py3-none-any.whl
# dist/docmore_paddleocr_vl_16-0.1.0.tar.gz
```

wheel 内路径确认是 `docmore/paddleocr_vl_16/...`（namespace 保留），不是
`paddleocr_vl_16/...`。`server/images.py`、`server/router.py` 均在 wheel 中。

### 3.4 真实端点探针

本机上游是 `http://127.0.0.1:8113/v1`。对外模型名契约是六个
`paddleocr-vl-1.6-*` 别名：`/v1/models` 只列这六个，请求里的 `model`
决定流水线模式，其他名字返回 404 `model_not_found`。真实探针通过
`paddleocr-vl-1.6-doc` 驱动两阶段流水线；`OPENAI_MODEL_NAME` 仍是上游
模型名，接入其他注册名的网关时可覆盖。

| 探针 | 输入 | 观察 | 验证点 |
| --- | --- | --- | --- |
| 端到端 CLI | 900×320 合成图片，两行英文 | 版面失败后 CPU 回退；Markdown 输出两行原文 | ONNX 版面、CPU 回退、裁剪、OCR prompt、Markdown 组装 |
| 表格 | 3×3 合成表格 | 裸 OTSL：`<fcel>Item<fcel>Qty...` | 无 wrapper 的 OTSL 解码 |
| 图表 | 4 柱合成柱状图 | Markdown 管道表（两列 `Quarter`、`Revenue` + 4 行） | 管道表 → HTML |
| Spotting | 同一柱状图 | 纯文本行（标题、数值、Q1–Q4） | 纯文本归一化 |
| Seal | 同一柱状图 | 纯文本行 | 纯文本归一化 |
| 本地 `/v1/models` | 无 | 六个别名，`backend` 为 `-doc` | 模型列表与 health 契约 |
| 本地 `/v1/parse` + `-table` | 3×3 表格图 | 单块 `type=table`，HTML 单元格正确，`model` 回显 | 整图单块、强制 `Table Recognition:` |
| 本地 `/v1/parse` + 未知 model | 任意图片 | 404 `model_not_found` | 模型名校验 |

端到端 CLI 运行日志里出现了预期的 CUDA 噪音：

```text
LoadLibrary failed for cudnn64_9.dll with error 2
```

随后流水线在锁内重建 CPU 会话并成功完成，Markdown 内容正确。这是
“GPU 版 onnxruntime 缺 cuDNN”的已知环境问题，不代表包本身失败。

### 3.5 未覆盖/未验证

- 真实发票、合同、扫描件、旋转/低质量页面上的版面与识别质量；
- 大规模多页文档的吞吐、内存、上游限流；
- 真正的 CUDA / DirectML 加速路径（只验证了缺 cuDNN 时的 CPU 降级）；
- Docling 生态下游（chunker/embedding）对导出文档的消费；
- 多页 URL 下载耗时与上游限流叠加后的真实吞吐；
- 图表识别返回 HTML 表格（实测当前端点是管道表）的具体变体。

## 4. 踩坑记录

### 4.1 端点只提供六个模型别名

**症状**：最初按“单一模型名 + 所有区域同一路径”实现，请求里传任意 model
都能通过；后续明确端点只提供六个 `paddleocr-vl-1.6-*` 别名，每个别名对应
一种流水线行为。

**结论/修复**：
- 新增 `mode.py`：`doc` 走完整两阶段，`table/text/formula/seal/chart`
  跳过版面、整页单块并强制对应 prompt；
- `/v1/models` 只列六个别名，`/health` 的 `backend` 默认 `-doc`；
- `/v1/parse` 与 `/v1/chat/completions` 校验 model，未知名字返回 404
  `model_not_found`；
- 服务端共享的 manager 固定 `doc` 模式，单块模式通过
  `BatchInputItem.prompt`/`prompt_type` 注入，避免为每个别名重建 ONNX 会话；
- 库调用仍允许自定义网关模型名，非别名回退 `doc`。

### 4.2 PaddleOCR-VL 的表格是裸 OTSL

**症状**：MiniCPM-OCR 习惯返回 `<otsl>...</otsl>`；PaddleOCR-VL 1.6 直接返回
`<fcel>...<nl>`。

**修复**：`html_table_from_raw()` 不要求 wrapper，检测到 OTSL token 就直接
解码；同时保留 `<otsl>` 去包装逻辑。

### 4.3 图表输出不是 HTML

**症状**：`Chart Recognition:` 实测返回 Markdown 管道表（`Quarter |
Revenue` + 数据行），而不是 HTML；某些图文场景可能只返回一句说明。

**修复**：
- 管道表走 `_pipe_table_to_html()`；
- 有 `<table>` 时透传；
- 都没有时作为普通文本输出，Markdown 侧 `chart` 分支按“表格/段落”区分；
- `/v1/parse` 的 chart 块只有在 content 含 `<table>` 时放 `content.html`，
  否则放 `content.text`。

### 4.4 行内公式用哪个 prompt

需求只给了 `Formula Recognition:`，没有单独的 inline 提示词。实现把
`inline_formula` 也送 `Formula Recognition:`，后处理剥掉 `\(...\)`，再按
label 渲染成行内 `$...$`。这样不引入需求外的 prompt，同时保留公式语义。

### 4.5 Spotting 没有对应版面类

PP-DocLayoutV3 的 25 类中没有 spotting。把普通 text 自动路由到
`Spotting:` 会改变输出语义，因此只在调用方显式
`prompt="Spotting:"`、`prompt_type="spotting"` 时使用。它仍然出现在
`TASK_PROMPTS` 和 `/v1/models` 的文档里，便于强制调用。

### 4.6 无版面回退 label 从 text 改为 ocr

MiniCPM 版本回退 label 是 `text`；PaddleOCR-VL 的任务名是 `ocr`。本包把
`PROMPT_LABEL` 设为 `ocr`，同时 `prompts.py` 把合成 label `ocr` 映射到
`OCR:`，避免“块 label 是 text、prompt 却是 OCR”的不一致。

### 4.7 继承的 `max_box_w` 坐标笔误

上游 paddleocr_vl 的 `layout.py` 计算 `max_box_w = x2 - y1`，本包在移植时
改为 `x2 - x1`，并保留 `_box_widths()` 与掩码回归测试。同类笔误
`_crop_from_bgr` 的 `xmin:ymax` 在当前实现中不可达（bgr 裁剪路径未启用），
未带入本包。

### 4.8 hatchling 把 namespace 拍平

若 `[tool.hatch.build.targets.wheel] packages =
["src/docmore/paddleocr_vl_16"]`，wheel 顶层会变成 `paddleocr_vl_16/`，
安装后 `import docmore.paddleocr_vl_16` 失败。必须写
`packages = ["src/docmore"]`，构建后用 zipfile 检查。

### 4.9 GPU 版 onnxruntime 缺 cuDNN

**症状**：首次真实运行，版面推理抛 `cudnn64_9.dll` 缺失，ONNX Runtime
直接报 `NOT_IMPLEMENTED`。

**修复**：`InferenceManager._detect_boxes()` 首次异常时，用
`CPUExecutionProvider` 重建会话重试一次；成功则继续，失败才标记
`_layout_failed` 并整页回退。日志里的 CUDA 噪音是预期现象。

### 4.10 测试运行方式与 extras

- Windows 上 `uv run pytest` 偶尔因 uv 启动器 trampoline 报错；用
  `uv run python -m pytest` 稳定。
- `uv run python -m pytest` 在默认依赖下是 72 passed / 10 skipped；看到
  skip 不能当成通过。发布前必须 `uv sync --all-extras`。

## 5. 设计理由备忘

| 选择 | 当时的理由 | 代价 |
| --- | --- | --- |
| 两阶段而不是整页 VLM | 表格/公式/图表需要不同任务提示词；版面本地跑更便宜 | 依赖 ONNX 模型和两次失败面 |
| 六个模型别名而不是单一 model | 端点契约要求按别名选择两阶段/整图单块；调用方只看模型名即可 | 非 doc 模式绕过版面，复杂页的阅读顺序不再由版面模型提供 |
| 版面 ONNX 本地跑 | 不占用 GPU 服务、延迟低、可离线 | 124 MB 模型分发问题；CPU 推理有耗时 |
| 每区域一个请求 | 输出格式可控，失败隔离 | 请求数多；上游吞吐成为瓶颈 |
| 贪心 + 升温度重试 | OCR 需要确定性，退化时才引入随机性 | 重试可能放大延迟 |
| 阅读顺序只做局部交换 | 全局重排风险高，可能破坏多栏 | 复杂版式仍可能顺序不佳 |
| `chunks` 作为唯一中间层 | 三种出口共享，测试可用 mock | 字段语义变更成本高 |
| 关闭识别时保留图片 | 不丢版面信息 | 输出中会出现“只有图没有字”的块 |
| 图表兼容表格与文本 | 实测两种输出都存在 | parse 的 chart content 有两个字段 |
| 资产键带页前缀 + `asset` 引用 | 多页/同尺寸图片不再靠猜 | 旧结果需要去重兼容层 |
| 流水线调用 `asyncio.to_thread` | 事件循环不被 OCR 拖住 | 线程数上限由 `MAX_CONCURRENT` 控制 |
| URL 源默认拒绝私网 | 避免把解析服务变成内网探测器 | 内网部署需显式开关 |
| 不支持的 ParseOptions 返回 400 | 契约诚实，调用方尽早发现 | 依赖“静默忽略”的旧调用方需调整 |

## 6. 已知限制与技术债

| # | 限制 | 影响 | 可能的处理 |
| --- | --- | --- | --- |
| 1 | HTML 输出等于 Markdown | 不是独立 HTML 文档 | 用 Markdown 渲染器生成完整 HTML |
| 2 | 页级/区域级线程池嵌套 | 极端页数下线程数放大 | 统一调度或共享线程池 |
| 3 | 没有跨页阅读顺序 | 跨页表格/续文顺序不保证 | 文档级后处理 |
| 4 | 真实文档未纳入自动回归 | 上游/模型变化无法自动发现 | 增加小型固定 fixture 与可选 live 测试 |
| 5 | 未验证 CUDA/DirectML 加速 | 性能未知 | 在目标机器上跑基准 |
| 6 | 文件类型识别回退魔数 | 非 PDF 的二进制可能误判 | 保持 Magika extra 安装 |
| 7 | URL 的 SSRF 防护是解析级而非连接级 | 理论上存在 DNS rebinding 时间窗 | 解析后固定 IP 再连接 |
| 8 | URL 取图整页串行 | 多页远程文档下载耗时叠加 | 并发预取并限流 |
| 9 | 图表识别不保留裁剪图 | 关闭识别才保留图片资产 | 需要时让 chunk 同时带 content 和 asset |
| 10 | `Spotting:` 无自动路由 | 调用方需显式 prompt | 增加 `SPOTTING_LABELS` 配置或专属模式 |

## 7. 后续改动的最小验证清单

任何触及流水线的改动，至少执行：

```bash
uv sync --all-extras
uv run python -m pytest -q          # 期望 117 passed
uv run python -m docmore.paddleocr_vl_16 --help
uv run python -m docmore.paddleocr_vl_16.server --help
```

触及版面/识别/后处理的改动，再加：

```text
[ ] 一份真实 PDF（含标题/正文/表格/图片）
[ ] 一份多页 PDF（检查 imgs/page_000N 与 Markdown 链接）
[ ] 一份表单或证件（检查上下顺序）
[ ] 一张表格图（检查 <table> 与 rowspan/colspan）
[ ] 一张图表（检查管道表/说明文字）
[ ] 一张公式图（检查 LaTeX 定界符）
[ ] 一次 spotting 强制 prompt
[ ] curl /health 看 layout_available
[ ] 一次 /v1/parse 看 blocks/assets/warnings
[ ] 一次 /v1/parse + pages/bbox 选项
[ ] 一次 /v1/parse + 公网 URL（或本地 URL + ALLOW_PRIVATE_URLS=1）
```

触及打包/依赖的改动，再加：

```text
[ ] uv build
[ ] 检查 wheel 内 docmore/paddleocr_vl_16/... 路径
[ ] 全新 venv 从 wheel 安装并 --help
[ ] Docker build + /health
```

## 8. 可以继续做的事

按价值排序：

1. 把真实文档做成可选 live 测试（默认跳过，CI 手动触发），覆盖多页 PDF、
   表格和图表；
2. 为多页 URL 做并发预取，并与 `MAX_PAGES`、外发连接数一起限流；
3. 增加吞吐/内存基准，确定 `MAX_WORKERS`、`MAX_CONCURRENT` 与上游并发的
   最佳组合；
4. URL 取图改为连接级 SSRF 防护（解析后固定 IP）；
5. 生成完整 HTML 文档而不是 Markdown 副本；
6. 统一页级/区域级线程池，避免极端页数的线程放大；
7. 为 `Spotting:` 设计显式模式/配置，而不是只靠强制 prompt。