# docmore-paddleocr-vl-16

PaddleOCR-VL 1.6 文档转换：PDF/图片 -> Markdown/HTML/JSON/Docling。

两阶段流水线，两个阶段使用独立的模型：

1. **版面检测** — PP-DocLayoutV3 ONNX（25 类版面元素），检测文本、标题、
   表格、公式、图表、印章等区域并给出阅读顺序（另对同列上下颠倒的
   文本块做局部修正）；
2. **区域识别** — 将每个区域裁剪后发送到 PaddleOCR-VL 1.6 的 OpenAI 兼容
   推理服务（默认 `http://127.0.0.1:8113/v1`，model
   `paddleocr-vl-1.6-doc`），按区域类型匹配任务提示词；
3. **后处理与组装** — OTSL / Markdown 表格 -> HTML、LaTeX 归一化、按阅读
   顺序组装 Markdown / 结构化 JSON（可选 DoclingDocument）。

namespace 包 `docmore.paddleocr_vl_16`，遵循 docmore 统一形式，复用
`docmore-utility` 做 PDF 渲染与文件类型识别。版面模型与
[docmore_paddleocr_vl](../docmore_paddleocr_vl) 完全相同（同一个
`models/PP-DocLayoutV3-onnx/inference.onnx`，25 类、CPU 可跑）。

## 文档

| 文档 | 内容 |
| --- | --- |
| [架构文档](docs/architecture.md) | 系统边界、两阶段数据流、扩展机制及存在理由 |
| [契约文档](docs/contracts.md) | Python/HTTP/上游/配置契约与兼容规则 |
| [说明文档](docs/guide.md) | 核心技术、安装、构建、发布 |
| [验证与记忆文档](docs/verification.md) | 实测记录、踩坑、设计理由、已知限制 |

## 模型别名与流水线模式

服务端（以及 `OPENAI_MODEL_NAME` 的默认值）只提供六个
`paddleocr-vl-1.6-*` 模型别名，每个别名选择一种流水线行为：

| model 名 | pipeline 行为 | 最终 prompt |
| --- | --- | --- |
| `paddleocr-vl-1.6-doc` | 完整两阶段（PP-DocLayoutV3 版面 + VLM） | 按块标签派生 |
| `paddleocr-vl-1.6-table` | 整图单块 | `Table Recognition:` |
| `paddleocr-vl-1.6-text` | 整图单块 | `OCR:` |
| `paddleocr-vl-1.6-formula` | 整图单块 | `Formula Recognition:` |
| `paddleocr-vl-1.6-seal` | 整图单块 | `Seal Recognition:`（不受 `USE_SEAL_RECOGNITION` 影响） |
| `paddleocr-vl-1.6-chart` | 整图单块 | `Chart Recognition:`（不受 `USE_CHART_RECOGNITION` 影响） |

`doc` 走版面检测；其余五个跳过版面，把整页作为一个块识别。HTTP 请求里的
`model` 必须命中这六个别名，否则返回 404
（`code: "model_not_found"`）。库调用时，`InferenceManager` 也接受
`mode="doc|table|text|formula|seal|chart"` 显式覆盖。

## 任务、提示词与输出

PaddleOCR-VL 1.6 的任务提示词：

| 任务 | 区域（PP-DocLayoutV3 label） | Prompt | 模型原始输出 | 最终 content |
| --- | --- | --- | --- | --- |
| 文本 | `text` / `doc_title` / `paragraph_title` / `figure_title` / `abstract` / `content` / `vertical_text` / `reference_content` / `vision_footnote` / `algorithm` / `formula_number` | `OCR:` | 纯文本 | 纯文本 |
| 表格 | `table` | `Table Recognition:` | 裸 OTSL 或 `<otsl>` 包装 | HTML `<table>` |
| 公式 | `display_formula` | `Formula Recognition:` | `\[ ... \]` / `\( ... \)` LaTeX | LaTeX（块级 `$$`） |
| 行内公式 | `inline_formula` | `Formula Recognition:` | `\( ... \)` LaTeX | LaTeX（行内 `$`） |
| 图表 | `chart` | `Chart Recognition:` | Markdown 管道表 / OTSL / 说明文字 | HTML `<table>` 或文本 |
| 印章 | `seal` | `Seal Recognition:` | 纯文本 | 纯文本 |
| 文本定位 | 无默认 label；强制 prompt 时可用 | `Spotting:` | 纯文本 | 纯文本 |
| 图片 | `image` / `header_image` / `footer_image` | — | 不识别 | 裁剪图资产 + Markdown 图片引用 |
| 页眉页脚等 | `header` / `footer` / `number` / `footnote` / `aside_text` / `reference` | — | 默认丢弃 | `INCLUDE_HEADERS_FOOTERS=true` 时按 `OCR:` 识别 |

表格 / 图表后处理顺序：已经是 HTML `<table>` 的原样保留；否则把 OTSL
token（`<fcel>` / `<ecel>` / `<lcel>` / `<ucel>` / `<xcel>` / `<nl>`）
解码成带 `rowspan` / `colspan` 的 HTML；再退化到 Markdown 管道表转换；
都不匹配时保留原始文本（图表只有说明文字时不会丢内容）。

> `Spotting:` 是 PaddleOCR-VL 1.6 的提示词之一，但 PP-DocLayoutV3 没有对应
> 版面类，因此默认流水线不会自动使用；调用方可以通过
> `BatchInputItem(prompt="Spotting:", prompt_type="spotting")` 强制整页
> 或指定区域执行。

## 版面模型目录

`LAYOUT_MODEL_DIR` 按以下顺序自动解析（第一个存在 `inference.onnx` 的目录）：

1. `<本项目>/models/PP-DocLayoutV3-onnx`
2. `<工作区>/docmore_paddleocr_vl/models/PP-DocLayoutV3-onnx`
3. `<工作区>/docmore_glmocr/models/PP-DocLayoutV3-onnx`

ONNX 文件约 124 MB，项目之间不复制。可用环境变量 `LAYOUT_MODEL_DIR`
显式指定。缺少 `onnxruntime`、模型文件不存在、检测异常或检出 0 区域时，
自动回退整页单区域 `OCR:`；也可用 `USE_LAYOUT_DETECTION=false` 强制关闭
版面检测。GPU 版 onnxruntime 缺少 cuDNN 时会在锁内重建 CPU 会话重试一次。

## 安装

```bash
uv sync --all-extras        # server + docling + layout + detect 全装
# 或 pip install "docmore-paddleocr-vl-16[server,docling,layout,detect]"
```

无 `[layout]` extra 时只做整页识别。服务端不需要 GPU：所有 OCR 推理都在
上游 PaddleOCR-VL 1.6 端点完成。

## 库用法

```python
from docmore.paddleocr_vl_16 import convert

convert("doc.pdf", "out/")                            # Markdown + HTML + JSON
convert("doc.pdf", "out/", output_format="docling")   # 额外输出 DoclingDocument
```

复用引擎实例处理多份文件：

```python
from docmore.paddleocr_vl_16 import InferenceManager, convert

manager = InferenceManager(max_workers=4)
convert("a.pdf", "out/", model=manager)
convert("b.pdf", "out/", model=manager)
```

## CLI

安装后提供三个命令：`docmore-paddleocr-vl-16`（转换）、`paddleocr-vl-16`
（转换别名）、`docmore-paddleocr-vl-16-serve`（HTTP 服务）。

```bash
docmore-paddleocr-vl-16 doc.pdf out/
paddleocr-vl-16 doc.pdf out/ --page-range 1-3
python -m docmore.paddleocr_vl_16 doc.pdf out/ --format docling --no-html
docmore-paddleocr-vl-16 ./docs out/ --max-workers 4 --batch-size 16
```

常用参数：`--page-range`、`--max-output-tokens`、`--max-workers`、
`--max-retries`、`--save-html/--no-html`、`--save-json/--no-json`、
`--batch-size`、`--paginate-output`、`--format default|docling`。

## 输出文件

每个输入文件写到 `out/<文件名>/`：

| 文件 | 说明 |
| --- | --- |
| `<name>.md` | 按阅读顺序组装的 Markdown |
| `<name>.json` | 结构化结果：`pages[].blocks[]`（label / bbox / content / reading_order / asset） |
| `<name>.html` | HTML（当前与 Markdown 内容一致；`--no-html` 关闭） |
| `<name>_metadata.json` | 页数、token 数、块/图片数量 |
| `imgs/page_0001/img_in_<label>_box_<x1>_<y1>_<x2>_<y2>.png` | 图片区域裁剪资产（按页分目录，避免同 bbox 覆盖） |
| `<name>.docling.json` / `.docling.md` | `--format docling` 时的 DoclingDocument |

多页输入时图片按 1-based 页目录分桶，`chunks[].asset` 记录图片块对应的
资产键；HTTP 的 `assets[]` 和 Docling 导出都按这个键精确取图。

## Server（OpenAI 兼容 + unified-parse）

```bash
docmore-paddleocr-vl-16-serve --host 0.0.0.0 --port 18910 --api-key secret
# 或: python -m docmore.paddleocr_vl_16.server
```

| 端点 | 说明 |
| --- | --- |
| `GET /health` | 健康检查（上游端点、`layout_available`） |
| `GET /v1/models` | 六个模型别名：`paddleocr-vl-1.6-doc/table/text/formula/seal/chart` |
| `POST /v1/chat/completions` | OpenAI 兼容：messages 内 image_url（data URI 或 URL）→ Markdown |
| `POST /v1/parse` | unified-parse/v1：content[] base64 / URL 图片 → pages + blocks + assets |

`/v1/parse` 的 `options` 支持 `pages`（如 `"1-3,5"`）、`include_images`、
`include_bounding_boxes`、`include_headers_footers`；不支持的取值返回 400
（`code: "unsupported_option"`）。URL 图片来源默认拒绝私网地址，内网图片
需设置 `PADDLEOCR_VL_16_SERVER_ALLOW_PRIVATE_URLS=1`。

服务配置环境变量前缀 `PADDLEOCR_VL_16_SERVER_`（HOST / PORT / API_KEYS /
MAX_PAGES / MAX_IMAGE_BYTES / MAX_CONCURRENT / PAGE_SEPARATOR /
ALLOW_URL_FETCH / ALLOW_PRIVATE_URLS / URL_FETCH_TIMEOUT /
URL_FETCH_MAX_REDIRECTS）。推理端点与版面配置沿用无前缀变量
（`OPENAI_API_BASE` / `OPENAI_MODEL_NAME` / `LAYOUT_MODEL_DIR` /
`USE_LAYOUT_DETECTION` / `LAYOUT_THRESHOLD` ...）。

## 配置

所有默认值可通过环境变量、`.env` 或 `local.env` 覆盖
（pydantic-settings，相对于**当前工作目录**查找）。优先级：

```text
真实环境变量  >  local.env  >  .env  >  代码默认值
```

常用变量：

```ini
OPENAI_API_BASE=http://127.0.0.1:8113/v1  # 裸地址 :8113 也会自动补 /v1
OPENAI_MODEL_NAME=paddleocr-vl-1.6-doc     # 六个别名之一，同时决定本地流水线模式
PIPELINE_MODE=                             # 可选：doc/table/text/formula/seal/chart
OPENAI_API_KEY=                           # 本地 llama.cpp 端点可留空
IMAGE_FORMAT=PNG                          # 也可 JPEG
MAX_OUTPUT_TOKENS=4096
TEMPERATURE=0.0
MAX_RETRIES=4
MAX_WORKERS=8

USE_LAYOUT_DETECTION=true
PROMPT_LABEL=ocr
LAYOUT_MODEL_DIR=          # 留空 = 自动解析；也可指定任意 PP-DocLayoutV3 目录
LAYOUT_TARGET_SIZE=800
LAYOUT_THRESHOLD=0.3
LAYOUT_NMS=true
LAYOUT_PROVIDER=auto       # auto / cpu / cuda / directml

INCLUDE_HEADERS_FOOTERS=false
USE_SEAL_RECOGNITION=true
USE_CHART_RECOGNITION=true
SHOW_FORMULA_NUMBER=true
PRETTIFY_MARKDOWN=false
```

服务端 URL 取图相关：

```ini
PADDLEOCR_VL_16_SERVER_ALLOW_URL_FETCH=true
PADDLEOCR_VL_16_SERVER_ALLOW_PRIVATE_URLS=false   # 内网图片需显式设为 true
PADDLEOCR_VL_16_SERVER_URL_FETCH_TIMEOUT=30
PADDLEOCR_VL_16_SERVER_URL_FETCH_MAX_REDIRECTS=5
```

> `LAYOUT_MODEL_DIR` 设为空字符串等价于不设置，会走上面的自动解析。

## 生成参数

客户端按 OpenAI Chat Completions 协议发请求：图片（data URI）在前、任务
提示词在后，不使用 system prompt。采样默认 `temperature=0.0`、
`top_p=1.0`；空输出或检测到退化重复 token 时升温度重试
（`min(T + 0.2 * attempt, 0.8)`），最多 `MAX_RETRIES + 1` 次。

## Docker

```bash
docker build -t docmore-paddleocr-vl-16:0.1.0 .
docker run -d -p 18910:18910 \
    -e OPENAI_API_BASE=http://<paddleocr-host>:8113/v1 \
    -v /path/to/PP-DocLayoutV3-onnx:/models/PP-DocLayoutV3-onnx:ro \
    -e LAYOUT_MODEL_DIR=/models/PP-DocLayoutV3-onnx \
    docmore-paddleocr-vl-16:0.1.0
```

版面 ONNX 模型不随镜像打包（体积原因），必须挂载或把
`USE_LAYOUT_DETECTION` 设为 `false`。

## 测试

```bash
uv sync --all-extras
uv run python -m pytest -q      # 117 passed
```

测试不依赖 GPU / 上游服务 / ONNX 模型：

- `tests/mock_server.py`：stdlib OpenAI 兼容 mock，按提示词返回
  text / OTSL 表格 / OTSL 图表 / LaTeX 公式 / 印章 / spotting 文本；
- `tests/test_pipeline.py`：`convert()` 端到端（Markdown / JSON / Docling）；
- `tests/test_layout_pipeline.py`：桩掉版面检测，验证区域路由、提示词、
  阅读顺序组装、图片资产、页号前缀、止损开关、强制 prompt、坐标笔误回归；
- `tests/test_mode.py`：六个模型别名、模式解析、整图单块绕过版面、
  印章/图表开关覆盖、显式 prompt 优先级；
- `tests/test_postprocess.py`：OTSL->HTML、HTML 包装、管道表兜底、公式
  定界符、图表说明文本、Markdown 渲染；
- `tests/test_parse_endpoint.py` / `test_server_api.py`：HTTP 路由、鉴权、
  parse 选项与资产匹配；
- `tests/test_server_concurrency.py`：并发请求不阻塞事件循环；
- `tests/test_url_sources.py`：URL 下载、重定向、大小限制与 SSRF 拒绝。

## 代码结构

```text
src/docmore/paddleocr_vl_16/
├── api.py            # convert() + 输出文件（md/html/json/docling）
├── cli.py            # docmore-paddleocr-vl-16
├── input.py          # PDF/图片 -> PIL（docmore-utility）
├── layout.py         # PP-DocLayoutV3 ONNX 检测 + 后处理（NMS/unclip/merge）
├── geometry.py       # 版面框重叠过滤
├── prompts.py        # label -> task -> prompt 路由
├── mode.py           # 六个模型别名 -> 流水线模式（doc/table/text/...）
├── postprocess.py    # OTSL->HTML、公式/文本归一化、Markdown 渲染
├── table.py          # OTSL -> HTML 解码器
├── model/
│   ├── __init__.py   # 两阶段编排（版面 -> 裁剪 -> 并行识别 -> 组装）
│   ├── client.py     # OpenAI 兼容 PaddleOCR-VL 1.6 客户端
│   └── util.py       # 退化重复检测
├── docling_bridge.py # chunks -> DoclingDocument
└── server/           # FastAPI：/v1/models /health /chat/completions /parse
```

`generate(batch) -> list[BatchOutputItem]` 是引擎与 CLI / server 之间的稳定
契约；`chunks` 使用 PP-DocLayoutV3 label（无版面回退时为 `ocr`），`content`
为纯文本 / LaTeX / HTML 表格（见上文任务表）。