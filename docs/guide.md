# 说明文档

> 适用版本：0.1.0 · 最后更新：2026-11-26
>
> 本文是面向使用者和发布者的总览：项目做什么、核心技术是什么、如何安装
> 使用、如何构建与发布。接口细节见 [契约文档](contracts.md)，架构与扩展
> 机制见 [架构文档](architecture.md)，实测与踩坑见
> [验证与记忆文档](verification.md)。

## 1. 项目定位

`docmore-paddleocr-vl-16` 把 PDF / 图片转换成 Markdown、HTML、结构化 JSON，
并可选导出 docling-core 的 `DoclingDocument`。

它遵循 docmore 系列的统一形式：

- namespace 包 `docmore.paddleocr_vl_16`，不会与工作区其他库冲突；
- 统一入口 `convert(input, output, ...)`；
- 引擎契约 `generate(batch) -> list[BatchOutputItem]`；
- 复用 `docmore-utility` 的 PDF 栅格化与文件类型识别；
- 可选 FastAPI 服务，同时提供 `unified-parse/v1` 与 OpenAI 兼容接口。

核心特点：**两阶段流水线，两个阶段使用独立模型**。

1. 版面检测：本地 PP-DocLayoutV3 ONNX，检测文本、标题、表格、公式、图表、
   印章等区域并给出阅读顺序；
2. 区域识别：把每个区域裁剪后发给 PaddleOCR-VL 1.6 的 OpenAI 兼容服务，
   按区域类型选择任务提示词；
3. 后处理：OTSL / 管道表 → HTML、LaTeX 归一化，按阅读顺序组装 Markdown /
   JSON。

## 2. 核心技术

### 2.1 为什么拆成两阶段

整页直接送 VLM 时，表格、公式、图表与正文混在一起，输出格式和阅读顺序
都难以控制。拆成两阶段后：

- 版面模型负责“哪里有什么、按什么顺序”，小而快，可在本地 CPU 跑；
- 识别模型只负责“这个区域里是什么”，按任务提示词输出对应格式；
- 两个模型可独立升级、独立部署，识别服务可远端扩容。

### 2.2 PP-DocLayoutV3 版面检测

- 输入统一缩放到 800×800，RGB 归一化，ONNX Runtime 推理；
- 输出 25 类版面元素（`text`、`doc_title`、`table`、`display_formula`、
  `inline_formula`、`chart`、`seal`、`image` 等）；
- 后处理包括：分数阈值、同类/异类 NMS、超大幅图片框过滤、containment
  合并、掩码转多边形、框扩张、按模型 order 排序；
- 再做一次重叠过滤和**局部阅读顺序修正**：只交换“上下堆叠、横向重叠
  ≥45%”的纯文本块，避免表单/证件页出现上下颠倒；
- 模型不可用、检测失败或 0 框时整页回退为单个 `ocr` 区域，不阻断流程；
  GPU 运行时失败会先在锁内重建 CPU 会话重试一次。

模型文件约 124 MB，项目之间不复制，默认自动查找同级
`docmore_paddleocr_vl` / `docmore_glmocr` 的模型目录。

### 2.3 PaddleOCR-VL 1.6 的模型别名与任务提示词

上游 PaddleOCR-VL 1.6 是任务提示型 OCR VLM。本包对外只暴露六个
`paddleocr-vl-1.6-*` 模型别名，每个别名选择一种流水线行为：

| model 名 | pipeline 行为 | 最终 prompt |
| --- | --- | --- |
| `paddleocr-vl-1.6-doc` | 完整两阶段（版面 + VLM） | 按块标签派生 |
| `paddleocr-vl-1.6-table` | 整图单块 | `Table Recognition:` |
| `paddleocr-vl-1.6-text` | 整图单块 | `OCR:` |
| `paddleocr-vl-1.6-formula` | 整图单块 | `Formula Recognition:` |
| `paddleocr-vl-1.6-seal` | 整图单块 | `Seal Recognition:` |
| `paddleocr-vl-1.6-chart` | 整图单块 | `Chart Recognition:` |

`doc` 模式按区域类型发送精确的任务串：

| 区域 | Prompt | 输出 |
| --- | --- | --- |
| 文本 / 标题 / 段落 | `OCR:` | 纯文本 |
| 表格 | `Table Recognition:` | OTSL / 管道表 → HTML |
| 独立公式 | `Formula Recognition:` | LaTeX |
| 行内公式 | `Formula Recognition:` | LaTeX（行内） |
| 图表 | `Chart Recognition:` | 表格或说明文字 |
| 印章 | `Seal Recognition:` | 纯文本 |
| 文本定位 | `Spotting:` | 纯文本（仅强制 prompt） |

请求形态是“图片 data URI 在前、提示词在后”，不使用 system prompt；默认
贪心解码（temperature 0.0）。空输出或退化重复时升温度重试。

### 2.4 OTSL / 表格后处理

实测 PaddleOCR-VL 1.6 的表格输出是裸 OTSL：

```text
<fcel>Item<fcel>Qty<fcel>Price<nl><fcel>Apple<fcel>3<fcel>12.5<nl>
```

`table.py` 把 `<fcel>`（填充单元格）、`<ecel>`（空）、`<lcel>`（左合并）、
`<ucel>`（上合并）、`<xcel>`（双向合并）、`<nl>`（换行）解码成带
`rowspan` / `colspan` 的 HTML `<table>`。后处理还兼容：

- 带或不带 `<otsl>` 包装；
- 模型直接返回 HTML 表（包括 `<html><body>` 包装）；
- 模型返回 Markdown 管道表（实测 `Chart Recognition:` 就是这种形态）；
- 以上都不匹配时保留纯文本，避免信息丢失。

### 2.5 公式与文本归一化

- 公式去掉 `$$...$$`、`\[...\]`、`\(...\)`、`$...$` 和 `equation` 等环境
  包装，存裸 LaTeX；Markdown 渲染时再加回块级/行内定界符；
- 文本统一 CRLF→LF，去掉代码围栏和首尾空白；
- 表格块原样输出 HTML，不再包代码围栏；
- 图表若只返回说明文字，作为普通段落输出。

### 2.6 并发与容错

- 页级线程池 + 区域级线程池，`MAX_WORKERS` 控制；
- 每区域最多重试 `MAX_RETRIES + 1` 次；空输出、退化重复、异常都会重试；
- 单区域失败只丢该块；整页失败才影响页级 `error`；
- 服务端 `MAX_CONCURRENT` 信号量限制同时处理的请求数；`generate()` 经
  `asyncio.to_thread` 执行，OCR 期间事件循环仍可响应 `/health`；
- 版面 ONNX 会话双重检查加锁，并发首请求只建一个会话；GPU 失败时锁内
  重建 CPU 会话并重试一次；
- URL 图片下载有超时、大小、重定向次数上限，并默认拒绝私网地址；
- 所有降级路径都保证“尽量输出”，而不是遇到问题就整份失败。

## 3. 安装

### 3.1 环境要求

- Python ≥ 3.10（推荐 3.12）；
- 上游 PaddleOCR-VL 1.6 OpenAI 兼容服务可达（默认
  `http://127.0.0.1:8113/v1`，模型名 `paddleocr-vl-1.6-doc`）；
- 版面检测需要 `inference.onnx`（可放在本项目 `models/`，或使用同级项目
  的副本，或用 `LAYOUT_MODEL_DIR` 指定）。

### 3.2 extras 选择

| 场景 | 安装 |
| --- | --- |
| 只要整页 OCR（无版面） | `pip install docmore-paddleocr-vl-16` |
| 完整两阶段（CPU） | `pip install "docmore-paddleocr-vl-16[layout,detect]"` |
| GPU 版面（CUDA） | `pip install "docmore-paddleocr-vl-16[layout-gpu,detect]"` |
| 导出 DoclingDocument | 再加 `[docling]` |
| 运行 HTTP 服务 | 再加 `[server]` |
| 全部 | `pip install "docmore-paddleocr-vl-16[layout,detect,docling,server]"` |

> `layout`（CPU）与 `layout-gpu`（CUDA）是二选一。`uv sync --all-extras`
> 会同时安装 `onnxruntime` 和 `onnxruntime-gpu`，只建议用于测试/开发。

开发环境：

```bash
cd docmore_paddleocr_vl_16
uv sync --all-extras
```

### 3.3 版面模型

默认解析顺序：

```text
<project>/models/PP-DocLayoutV3-onnx
<workspace>/docmore_paddleocr_vl/models/PP-DocLayoutV3-onnx
<workspace>/docmore_glmocr/models/PP-DocLayoutV3-onnx
```

第一个包含 `inference.onnx` 的目录生效。也可显式指定：

```bash
export LAYOUT_MODEL_DIR=/path/to/PP-DocLayoutV3-onnx
```

若完全不做版面检测：

```bash
export USE_LAYOUT_DETECTION=false
```

## 4. 使用

### 4.1 CLI

安装后提供三个命令：`docmore-paddleocr-vl-16`（转换）、`paddleocr-vl-16`
（转换别名）、`docmore-paddleocr-vl-16-serve`（HTTP 服务）。

```bash
docmore-paddleocr-vl-16 doc.pdf out/
python -m docmore.paddleocr_vl_16 doc.pdf out/ --page-range 1-3
docmore-paddleocr-vl-16 doc.pdf out/ --format docling --no-html
docmore-paddleocr-vl-16 ./docs out/ --max-workers 4 --batch-size 16
```

参数：

| 参数 | 说明 |
| --- | --- |
| `--page-range` | PDF 页码，1 起，如 `1-5,7` |
| `--max-output-tokens` | 单区域输出 token 上限 |
| `--max-workers` | 并发线程数 |
| `--max-retries` | 单区域重试次数 |
| `--save-html/--no-html` | 是否写 `.html` |
| `--save-json/--no-json` | 是否写结构化 `.json` |
| `--batch-size` | 每批页数（默认 28） |
| `--paginate-output` | 多页之间插入分页标记 |
| `--format` | `default` 或 `docling` |

### 4.2 库

```python
from docmore.paddleocr_vl_16 import convert

convert("doc.pdf", "out/")
convert("doc.pdf", "out/", output_format="docling", save_html=False)
```

复用引擎实例处理多份文件：

```python
from docmore.paddleocr_vl_16 import InferenceManager, convert

manager = InferenceManager(max_workers=4)
convert("a.pdf", "out/", model=manager)
convert("b.pdf", "out/", model=manager)
```

强制整页任务（例如只用表格识别）：

```python
from PIL import Image
from docmore.paddleocr_vl_16 import InferenceManager
from docmore.paddleocr_vl_16.schema import BatchInputItem

manager = InferenceManager()
out = manager.generate([
    BatchInputItem(
        image=Image.open("table.png"),
        prompt="Table Recognition:",
        prompt_type="table",
    )
])[0]
print(out.markdown)
```

文本 spotting：

```python
out = manager.generate([
    BatchInputItem(
        image=Image.open("page.png"),
        prompt="Spotting:",
        prompt_type="spotting",
    )
])[0]
```

### 4.3 HTTP 服务

```bash
docmore-paddleocr-vl-16-serve --host 0.0.0.0 --port 18910 --api-key secret
# 或
python -m docmore.paddleocr_vl_16.server --port 18910
```

健康检查：

```bash
curl http://127.0.0.1:18910/health
```

OpenAI 兼容调用：

```bash
curl http://127.0.0.1:18910/v1/chat/completions \
  -H "Authorization: Bearer secret" \
  -H "Content-Type: application/json" \
  -d '{"model":"paddleocr-vl-1.6-doc","messages":[{"role":"user","content":[
        {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}
      ]}]}'
```

统一解析接口：

```bash
curl http://127.0.0.1:18910/v1/parse \
  -H "Authorization: Bearer secret" \
  -H "Content-Type: application/json" \
  -d '{"content":[{"type":"image","source":{"type":"base64","data":"..."}}],
       "options":{"include_images":true,"pages":"1-3","include_bounding_boxes":false}}'
```

远程图片：

```bash
curl http://127.0.0.1:18910/v1/parse \
  -H "Content-Type: application/json" \
  -d '{"content":[{"type":"image","source":{"type":"url",
       "url":"https://example.com/scan.png"}}]}'
```

生产环境注意 URL 取图的默认防护：`ALLOW_PRIVATE_URLS=false` 时拒绝内网
地址；内网图片需要显式设置
`PADDLEOCR_VL_16_SERVER_ALLOW_PRIVATE_URLS=1`，或整体用
`PADDLEOCR_VL_16_SERVER_ALLOW_URL_FETCH=0` 关闭。下载超时、重定向次数和
大小上限分别由 `URL_FETCH_TIMEOUT`、`URL_FETCH_MAX_REDIRECTS`、
`MAX_IMAGE_BYTES` 控制。

完整字段见 [契约文档第 5 节](contracts.md#5-http-服务契约)。

### 4.4 Docker

```bash
docker build -t docmore-paddleocr-vl-16:0.1.0 .
docker run -d --name paddleocr-vl-16 \
    -p 18910:18910 \
    -e OPENAI_API_BASE=http://<paddleocr-host>:8113/v1 \
    -v /path/to/PP-DocLayoutV3-onnx:/models/PP-DocLayoutV3-onnx:ro \
    -e LAYOUT_MODEL_DIR=/models/PP-DocLayoutV3-onnx \
    docmore-paddleocr-vl-16:0.1.0
```

镜像基于 `python:3.12-slim`，安装
`[server,detect,docling,layout]`，以非 root 用户运行，暴露 18910。
ONNX 模型不随镜像打包，必须挂载或关闭版面检测。

## 5. 输出文件

每个输入文件写到 `out/<文件名>/`：

| 文件 | 说明 |
| --- | --- |
| `<name>.md` | 按阅读顺序组装的 Markdown |
| `<name>.html` | 当前与 Markdown 内容相同 |
| `<name>.json` | 结构化：`pages[].blocks[]`（label/bbox/content/reading_order/asset） |
| `<name>_metadata.json` | 页数、token、块数、图片数 |
| `imgs/page_0001/img_in_<label>_box_<x1>_<y1>_<x2>_<y2>.png` | 图片区域裁剪（按 1-based 页目录分桶） |
| `<name>.docling.json` / `.docling.md` | `--format docling` 时 |

结构化 JSON 示例：

```json
{
  "file_name": "doc.pdf",
  "model": "paddleocr-vl-1.6-doc",
  "num_pages": 1,
  "total_token_count": 123,
  "pages": [
    {
      "page_num": 0,
      "page_box": [0, 0, 1000, 1400],
      "markdown": "……",
      "blocks": [
        {
          "id": "p0-b0",
          "reading_order": 0,
          "label": "text",
          "bbox": [50, 160, 950, 300],
          "content": "……"
        }
      ]
    }
  ]
}
```

## 6. 配置摘要

全部变量与优先级见 [契约文档第 7 节](contracts.md#7-配置契约)。最常用的：

```ini
OPENAI_API_BASE=http://127.0.0.1:8113/v1
OPENAI_MODEL_NAME=paddleocr-vl-1.6-doc   # 六个别名之一，同时决定流水线模式
PIPELINE_MODE=                           # 可选显式覆盖：doc/table/text/formula/seal/chart
OPENAI_API_KEY=

USE_LAYOUT_DETECTION=true
LAYOUT_MODEL_DIR=
LAYOUT_PROVIDER=auto
LAYOUT_THRESHOLD=0.3

MAX_WORKERS=8
MAX_RETRIES=4
TEMPERATURE=0.0
```

服务端配置统一使用 `PADDLEOCR_VL_16_SERVER_` 前缀；流水线配置不带前缀。
`.env` / `local.env` 按**当前工作目录**查找。

## 7. 构建

### 7.1 本地构建

```bash
uv sync --all-extras
uv build
```

产物：

```text
dist/docmore_paddleocr_vl_16-0.1.0-py3-none-any.whl
dist/docmore_paddleocr_vl_16-0.1.0.tar.gz
```

构建后端是 hatchling。`pyproject.toml` 中 wheel 的 package 根必须是
`src/docmore`（namespace 根），不能写成 `src/docmore/paddleocr_vl_16`——
后者会被拍平，安装后 `import docmore.paddleocr_vl_16` 失败。

### 7.2 校验 wheel 的 namespace

```bash
uv run python - <<'PY'
import glob, zipfile

wheel = glob.glob("dist/*.whl")[0]
names = zipfile.ZipFile(wheel).namelist()
assert "docmore/paddleocr_vl_16/__init__.py" in names, "namespace 被拍平"
assert "docmore/paddleocr_vl_16/server/router.py" in names
print("wheel OK:", wheel, f"{len(names)} entries")
PY
```

### 7.3 全新环境冒烟

```bash
uv venv .smoke-venv
# Windows
uv pip install --python .smoke-venv/Scripts/python.exe dist/docmore_paddleocr_vl_16-0.1.0-py3-none-any.whl
.smoke-venv/Scripts/python.exe -m docmore.paddleocr_vl_16 --help
# Linux/macOS 把 Scripts 换成 bin
rm -rf .smoke-venv
```

### 7.4 测试

```bash
uv sync --all-extras
uv run python -m pytest -q
```

- 全 extras：117 passed；
- 测试不需要 GPU、上游服务或 ONNX 模型：`tests/mock_server.py` 提供
  OpenAI 兼容 mock，版面检测在测试中被禁用或桩掉；URL 测试用本地
  `ThreadingHTTPServer` 提供图片。

建议使用 `uv run python -m pytest` 而不是 `uv run pytest`（Windows 上
uv 生成的 pytest 启动器偶尔有 trampoline 问题）。

## 8. 发布

### 8.1 版本号

需要同步两处：

- `pyproject.toml` 的 `[project].version`；
- `src/docmore/paddleocr_vl_16/__init__.py` 的 `__version__`。

服务端 `/health`、OpenAPI 标题都读取 `__version__`。发布前确认 README
和 docs 的“适用版本”一致。

### 8.2 发布前检查清单

```text
[ ] uv sync --all-extras
[ ] uv run python -m pytest -q          # 117 passed
[ ] uv build
[ ] wheel namespace 校验（7.2）
[ ] 全新 venv 冒烟（7.3）
[ ] docmore-paddleocr-vl-16 --help / serve --help
[ ] 有 ONNX 模型时跑一份真实 PDF（发票/表格各一）
[ ] 有上游服务时 curl /health 与 /v1/parse
[ ] 版本号两处一致
[ ] README / docs 更新
```

### 8.3 发布到 PyPI

```bash
rm -rf dist/
uv build

uvx twine check dist/*
uv publish --token "$PYPI_TOKEN" dist/*
```

注意：

- `pyproject.toml` 里的 `[[tool.uv.index]]` 指向清华镜像，仅用于**安装**
  依赖；`uv publish` 默认发布到 PyPI；
- 发布前先在干净 venv 里从 `dist/` 安装并跑 `--help`；
- 当前 sdist 只显式 `include = ["src/docmore"]`；若希望 `docs/` 随包
  分发，需要把 `docs` 加入 `[tool.hatch.build.targets.sdist]` 的 include。

### 8.4 发布 Docker 镜像

```bash
docker build -t docmore-paddleocr-vl-16:0.1.0 .
docker tag docmore-paddleocr-vl-16:0.1.0 <registry>/docmore-paddleocr-vl-16:0.1.0
docker push <registry>/docmore-paddleocr-vl-16:0.1.0
```

## 9. 目录结构

```text
docmore_paddleocr_vl_16/
├── docs/
│   ├── architecture.md      # 架构：边界、数据流、扩展机制
│   ├── contracts.md         # 契约：API、上游协议、配置、兼容
│   ├── guide.md             # 说明：技术、安装、构建、发布
│   └── verification.md      # 验证：踩坑、设计理由、开发过程
├── src/docmore/paddleocr_vl_16/
│   ├── api.py               # convert + 输出文件
│   ├── cli.py               # CLI
│   ├── input.py             # PDF/图片 -> PIL
│   ├── settings.py          # 流水线配置
│   ├── layout.py            # PP-DocLayoutV3 ONNX + 后处理
│   ├── geometry.py          # 版面框过滤
│   ├── prompts.py           # label -> task -> prompt
│   ├── postprocess.py       # 归一化 + Markdown
│   ├── table.py             # OTSL -> HTML
│   ├── docling_bridge.py    # DoclingDocument
│   ├── schema.py            # 数据类
│   ├── model/
│   │   ├── __init__.py      # 两阶段编排
│   │   ├── client.py        # OpenAI 客户端
│   │   └── util.py          # 退化重复检测
│   └── server/              # FastAPI
├── tests/
├── Dockerfile
├── pyproject.toml
└── README.md
```

## 10. 文档索引

- [README](../README.md)：快速上手与配置速查；
- [架构文档](architecture.md)：系统边界、数据流、扩展机制及存在理由；
- [契约文档](contracts.md)：Python/HTTP/上游/配置契约与兼容规则；
- [验证与记忆文档](verification.md)：实测记录、踩坑、设计理由、开发过程。