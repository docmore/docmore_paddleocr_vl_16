# 契约文档

> 适用版本：0.1.0 · 最后更新：2026-11-26
>
> 本文记录对外承诺：Python API、引擎契约、上游 OpenAI 协议、HTTP
> unified-parse/v1、配置变量与兼容规则。架构与数据流见
> [架构文档](architecture.md)，使用/构建步骤见 [说明文档](guide.md)。

## 1. Python 库契约

### 1.1 入口

```python
from docmore.paddleocr_vl_16 import convert

convert(
    input_path: str | Path,
    output_path: str | Path,
    *,
    page_range: str | None = None,
    max_output_tokens: int | None = None,
    max_workers: int | None = None,
    max_retries: int | None = None,
    save_html: bool = True,
    save_json: bool = True,
    batch_size: int | None = None,
    paginate_output: bool = False,
    output_format: str = "default",   # "default" | "docling"
    model=None,
    progress: Callable[[str], None] = print,
) -> Path
```

- `input_path` 可以是单个 PDF/图片，或包含多个受支持文件的目录。
- 输出写到 `output_path/<文件名>/`；返回该每文件目录。
- `page_range` 仅对 PDF 生效，1 起，支持 `"1-5,7,9-12"`；越界/空范围报错。
- `output_format="docling"` 额外写 `<name>.docling.json` / `.docling.md`，
  需要安装 `[docling]`（`docling-core`）；缺少时只打印警告，不影响默认输出。
- `model` 传入一个已有的 `InferenceManager` 可复用 ONNX 会话与 HTTP 客户端；
  `InferenceManager(mode=...)` 可用 `doc/table/text/formula/seal/chart`
  显式选择流水线模式（默认从模型别名推导）。

支持的扩展名：`.pdf .png .jpg .jpeg .gif .webp .tiff .bmp`（大小写均可）。
单文件扩展名不受支持时抛 `ValueError`。

### 1.2 引擎契约（稳定）

```python
manager.generate(batch: list[BatchInputItem]) -> list[BatchOutputItem]
```

- 输入顺序与输出顺序一一对应；
- 单页失败不会抛异常，而是返回 `error=True` 的页结果；
- `generate()` 是同步、线程安全的；CLI 和服务端都只依赖这个契约；
- 直接调用时 `BatchInputItem.page_index` 可省略，此时图片资产使用旧的
  `imgs/<name>.png` 键；`convert()` 和 HTTP 服务始终带页号。

`BatchInputItem`：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `image` | `PIL.Image.Image` | 页面图 |
| `prompt` | `str \| None` | 强制整页单任务提示词；设置后跳过版面检测 |
| `prompt_type` | `str \| None` | 强制模式下用于后处理的 label/task，如 `text`、`inline_formula`、`spotting` |
| `page_index` | `int \| None` | 0 起页号，进入图片资产键 |
| `include_headers_footers` | `bool \| None` | 覆盖 manager 级页眉页脚开关 |

`BatchOutputItem`：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `markdown` | `str` | 本页阅读顺序 Markdown |
| `html` | `str` | 当前与 Markdown 相同 |
| `chunks` | `list[dict]` | `{"label", "bbox", "content", "asset"?}` |
| `raw` | `str` | `<label>\ncontent` 调试串 |
| `page_box` | `[x0,y0,x1,y1]` | 本页像素范围 |
| `token_count` | `int` | 上游 completion tokens |
| `images` | `dict[str, PIL.Image]` | 资产键 → 裁剪图 |
| `error` | `bool` | 整页无 Markdown 且有识别失败时为 true |

`chunks[].label` 是 PP-DocLayoutV3 原始 label；无版面回退时为 `ocr`。
`chunks[].content` 的形态：

| label / task | content |
| --- | --- |
| 文本类（`text`、`ocr`、标题、段落…） | 纯文本 |
| `table` | HTML `<table>` |
| `chart` | HTML `<table>` 或纯文本说明 |
| `display_formula` / `inline_formula` | 裸 LaTeX |
| `seal` / `spotting` | 纯文本 |
| 图片类 | `""`，同时有 `asset` 指向 `images` 中的键 |

## 2. 模型别名与提示词路由契约

### 2.1 六个模型别名

端点只接受下列六个 `paddleocr-vl-1.6-*` 模型名：

| model 名 | pipeline 行为 | 最终 prompt |
| --- | --- | --- |
| `paddleocr-vl-1.6-doc` | 完整两阶段（版面 + VLM） | 按块标签派生 |
| `paddleocr-vl-1.6-table` | 整图单块 | `Table Recognition:` |
| `paddleocr-vl-1.6-text` | 整图单块 | `OCR:` |
| `paddleocr-vl-1.6-formula` | 整图单块 | `Formula Recognition:` |
| `paddleocr-vl-1.6-seal` | 整图单块 | `Seal Recognition:` |
| `paddleocr-vl-1.6-chart` | 整图单块 | `Chart Recognition:` |

- `doc` 之外的模式跳过版面检测，把整页作为单块，并强制对应 prompt；
- `-seal` / `-chart` 的强制 prompt 优先于 `USE_SEAL_RECOGNITION` /
  `USE_CHART_RECOGNITION` 开关；
- 六个别名以外的 model 在 HTTP 层返回 404
  （`code: "model_not_found"`）；库调用 `InferenceManager` 时非别名仍回退
  到 `doc`，便于接入自定义网关；
- `BatchInputItem.prompt` / `prompt_type` 优先级最高，可覆盖模式。

### 2.2 提示词路由

任务提示词（PaddleOCR-VL 1.6）：

```python
TASK_PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "inline_formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "spotting": "Spotting:",
    "seal": "Seal Recognition:",
}
```

label → task：

| task | PP-DocLayoutV3 label |
| --- | --- |
| `ocr` | `text`、`doc_title`、`paragraph_title`、`figure_title`、`abstract`、`content`、`vertical_text`、`reference_content`、`vision_footnote`、`algorithm`、`formula_number`，以及合成 label `ocr` |
| `table` | `table` |
| `formula` | `display_formula`、合成 label `formula` |
| `inline_formula` | `inline_formula` |
| `chart` | `chart` |
| `seal` | `seal` |
| `spotting` | 合成 label `spotting`（仅强制 prompt） |
| skip | `image`、`header_image`、`footer_image` |
| abandon | `header`、`footer`、`number`、`footnote`、`aside_text`、`reference` |

- `USE_SEAL_RECOGNITION=false` 时 seal 不送模型，改为图片资产；
- `USE_CHART_RECOGNITION=false` 时 chart 不送模型，改为图片资产；
- `INCLUDE_HEADERS_FOOTERS=true` 时 abandon 类改用 `OCR:`；
- 未知 label 回退 `ocr`；
- 调用方显式给 `prompt` 时，`prompt_type` 优先决定后处理任务；
  只有 `prompt_type` 缺失时才从 prompt 反查。

`Spotting:` 没有默认版面路由，因为 PP-DocLayoutV3 不含 spotting 类。
需要时：

```python
BatchInputItem(
    image=page,
    prompt="Spotting:",
    prompt_type="spotting",
)
```

## 3. 上游 OpenAI 协议契约

- 端点：`POST {OPENAI_API_BASE}/chat/completions`；
- `OPENAI_API_BASE` 可以是裸地址（`http://host:8113`），客户端只在 URL
  path 为空时追加 `/v1`；已有路径原样保留；
- 模型名：`OPENAI_MODEL_NAME`，默认 `paddleocr-vl-1.6-doc`（六个
  `paddleocr-vl-1.6-*` 别名之一）；它是上游模型名，同时决定本地流水线
  模式。接入使用其他注册名的网关时可用环境变量覆盖，此时本地默认走
  `doc` 两阶段流水线；
- 消息形态：单条 user 消息，content 数组 **图片 data URI 在前、文字 prompt
  在后**，不使用 system prompt；
- 图片编码：`IMAGE_FORMAT`（默认 `PNG`，也可 `JPEG`）；
- 采样：`temperature=0.0`、`top_p=1.0` 起始；
- 空输出 / 退化重复 / 异常：升温度重试，`min(T + 0.2×attempt, 0.8)`，
  `top_p=0.95`，最多 `MAX_RETRIES + 1` 次；
- 上游返回的 `usage.completion_tokens` 汇总到 `token_count`；
- 本地 llama.cpp 端点允许空 API key；客户端在 key 为空时发送 `EMPTY`。

期望的原始输出：

| Prompt | 期望 | 也接受 |
| --- | --- | --- |
| `OCR:` | 纯文本 | Markdown 行内公式 |
| `Table Recognition:` | 裸 OTSL 或 `<otsl>` | HTML `<table>`、Markdown 管道表 |
| `Formula Recognition:` | `\[...\]` / `\(...\)` / `$$...$$` | 裸 LaTeX |
| `Chart Recognition:` | Markdown 管道表 / OTSL / 说明文字 | HTML `<table>` |
| `Seal Recognition:` | 纯文本 | — |
| `Spotting:` | 纯文本 | — |

后处理永远保留纯文本兜底：表格解析失败时不会丢弃模型输出。

## 4. 输出文件契约

`save_merged_output(output_dir, file_name, results, ...)` 写：

| 文件 | 内容 |
| --- | --- |
| `<name>.md` | 多页 Markdown 拼接；`paginate_output` 时插页分隔 |
| `<name>.html` | 当前与 Markdown 内容一致（`save_html=False` 时不写） |
| `<name>.json` | `{file_name, model, num_pages, total_token_count, pages[]}` |
| `<name>_metadata.json` | 每页 token/块/图片统计 |
| `imgs/page_0001/...png` | 图片资产（1 起页目录） |
| `<name>.docling.json/.md` | `output_format="docling"` 时 |

结构化 JSON 的页：

```json
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
```

图片块额外带 `asset`（`imgs/page_0001/...`），与 `result.images` 的键一致。
旧格式的无页号资产键在写入时会被去重改写，并同步修改 Markdown 链接。

## 5. HTTP 服务契约

启动：

```bash
docmore-paddleocr-vl-16-serve --host 0.0.0.0 --port 18910 --api-key secret
# 或 python -m docmore.paddleocr_vl_16.server
```

- 默认端口 `18910`；
- `API_KEYS` 非空时要求 `Authorization: Bearer <key>`；为空表示开放；
- 所有错误使用 OpenAI 错误信封：
  `{"error": {"message", "type", "param", "code"}}`。

### 5.1 `GET /health`

```json
{
  "status": "ok",
  "backend": "paddleocr-vl-1.6-doc",
  "upstream": "http://127.0.0.1:8113/v1",
  "layout_available": true
}
```

### 5.2 `GET /v1/models`

返回六个别名：`paddleocr-vl-1.6-doc`、`-table`、`-text`、`-formula`、
`-seal`、`-chart`。请求里的 `model` 决定流水线模式；其他 model 返回 404
（`code: "model_not_found"`）。

### 5.3 `POST /v1/chat/completions`

OpenAI 兼容形态：最后一条 user 消息里的每个 `image_url` 是一页；
data URI / http(s) URL 都可。返回拼接后的 Markdown。

```json
{
  "model": "paddleocr-vl-1.6-doc",
  "messages": [
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]}
  ]
}
```

部分页失败时返回 200，并带 `X-PaddleocrVl16-Partial: 1`；全部失败或无任何
识别结果返回 502。

### 5.4 `POST /v1/parse`

请求：

```json
{
  "model": "paddleocr-vl-1.6-doc",
  "content": [
    {"type": "image", "source": {"type": "base64", "data": "..."}},
    {"type": "image", "source": {"type": "url", "url": "https://..."}}
  ],
  "options": {
    "pages": "1-3",
    "include_images": true,
    "include_bounding_boxes": true,
    "include_headers_footers": false
  }
}
```

也接受 legacy `messages[]`（取 user 的 image_url）。

支持的 options：

| 字段 | 默认 | 行为 |
| --- | --- | --- |
| `pages` | 全部 | 1 起页范围，结果保留原始 `index`/`number` |
| `include_images` | `true` | false 时不内联 base64，并删除页 Markdown 图片链接 |
| `include_bounding_boxes` | `true` | false 时去掉 block/asset 的 bbox |
| `include_headers_footers` | `false` | 传给每页 |
| `output_format` | `"markdown"` | 其他值 → 400 |
| `table_format` | `"html"` | 其他值 → 400 |
| `language` | `"auto"` | 其他值 → 400 |
| `include_lines` | `false` | true → 400（未实现行级） |
| `response_schema` | `null` | 非 null → 400 |
| `ocr_options` | `null` | 非空时忽略并加 warning |

响应遵循 unified-parse/v1：

```json
{
  "id": "ocr-...",
  "object": "parse.result",
  "model": "paddleocr-vl-1.6-doc",
  "schema": "unified-parse/v1",
  "pages": [
    {
      "index": 0,
      "number": 1,
      "size": {"width": 1000, "height": 1400, "unit": "px", "dpi": 200},
      "markdown": "……",
      "blocks": [
        {"id": "p0-b0", "type": "paragraph", "reading_order": 0,
         "content": {"text": "……"}, "bbox": [50, 160, 950, 300]}
      ]
    }
  ],
  "assets": [
    {"id": "p0-b1", "type": "image", "page_index": 0,
     "bbox": [650, 700, 950, 880], "base64": "...", "mime_type": "image/jpeg"}
  ],
  "content": "多页 Markdown 以 --- 分隔",
  "usage": {"prompt_tokens": 0, "completion_tokens": 123,
            "total_tokens": 123, "pages_processed": 1},
  "warnings": []
}
```

block 类型映射：`text/ocr/content/vertical_text → paragraph`，标题 → `title`
（`doc_title` level 1，`paragraph_title/figure_title` level 2），
`table → table`，`chart → chart`，公式 → `formula`，`seal → seal`，
图片 → `image`。`chart` 内容若为 `<table>` 放 `content.html`，否则放
`content.text`。

## 6. 图片资产契约

- 键格式：`imgs/page_%04d/img_in_<label>_box_<x1>_<y1>_<x2>_<y2>.png`
  （页号 1 起）；未提供 `page_index` 时为 `imgs/img_in_....png`；
- `chunk.asset` 是 `BatchOutputItem.images` 的精确键；
- `/v1/parse` 的 `assets[]` 以 block id 关联，优先按 `asset` 取图；
- Docling 导出同样优先按 `asset` 取图，旧结果才回退到相同尺寸匹配；
- `include_images=false` 时 `assets[].base64` 为 `null`，Markdown 图片链接被删。

## 7. 配置契约

优先级：真实环境变量 > `local.env` > `.env` > 代码默认值。
文件按**当前工作目录**查找。

流水线变量（无前缀）：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `OPENAI_API_BASE` | `http://127.0.0.1:8113/v1` | 上游地址，裸地址自动补 `/v1` |
| `OPENAI_MODEL_NAME` | `paddleocr-vl-1.6-doc` | 上游模型名，同时决定流水线模式 |
| `PIPELINE_MODE` | `""` | 显式覆盖模式：`doc/table/text/formula/seal/chart` |
| `OPENAI_API_KEY` | `""` | 本地端点可空 |
| `IMAGE_FORMAT` | `PNG` | `PNG` / `JPEG` |
| `IMAGE_DPI` | `200` | PDF 渲染 DPI |
| `MIN_PDF_IMAGE_DIM` | `1024` | PDF 短边最小像素 |
| `MIN_IMAGE_DIM` | `1536` | 独立图片短边放大阈值 |
| `MAX_OUTPUT_TOKENS` | `4096` | 单区域输出上限 |
| `TEMPERATURE` | `0.0` | 基础温度 |
| `TOP_P` | `1.0` | 基础 top_p |
| `MAX_RETRIES` | `4` | 单区域重试次数 |
| `MAX_WORKERS` | `8` | 页/区域线程池 |
| `TIMEOUT` | `300` | 单请求超时秒 |
| `USE_LAYOUT_DETECTION` | `true` | 关闭后用整页 `ocr` |
| `PROMPT_LABEL` | `ocr` | 无版面回退 label |
| `LAYOUT_MODEL_DIR` | 自动解析 | 含 `inference.onnx` 的目录 |
| `LAYOUT_TARGET_SIZE` | `800` | 版面模型输入边长 |
| `LAYOUT_THRESHOLD` | `0.3` | 检测分数阈值 |
| `LAYOUT_NMS` | `true` | 分类 NMS |
| `LAYOUT_UNCLIP_RATIO` | `(1.0, 1.0)` | 框扩张比例 |
| `LAYOUT_PROVIDER` | `auto` | `auto/cpu/cuda/directml` |
| `INCLUDE_HEADERS_FOOTERS` | `false` | 页眉页脚等 |
| `USE_SEAL_RECOGNITION` | `true` | 印章识别 |
| `USE_CHART_RECOGNITION` | `true` | 图表识别 |
| `SHOW_FORMULA_NUMBER` | `true` | 公式编号 |
| `PRETTIFY_MARKDOWN` | `false` | 合并多余空行 |

服务端变量（前缀 `PADDLEOCR_VL_16_SERVER_`）：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `18910` | 监听端口 |
| `API_KEYS` | `[]` | Bearer token 列表，空=开放 |
| `MAX_PAGES` | `32` | 单请求最大图片数 |
| `MAX_IMAGE_BYTES` | `26214400` | 解码后单图上限（25 MiB） |
| `MAX_CONCURRENT` | `4` | 同时处理的请求数 |
| `PAGE_SEPARATOR` | `"\n\n"` | chat 多页分隔 |
| `ALLOW_URL_FETCH` | `true` | 是否接受 URL 来源 |
| `ALLOW_PRIVATE_URLS` | `false` | 是否允许私网 URL |
| `URL_FETCH_TIMEOUT` | `30` | 下载超时秒 |
| `URL_FETCH_MAX_REDIRECTS` | `5` | 最大重定向次数 |

`LAYOUT_MODEL_DIR` 设为空字符串等价于自动解析。

## 8. 版本与兼容

- 当前版本 `0.1.0`；`__version__` 与 `pyproject.toml` 必须同步。
- `generate(batch) -> list[BatchOutputItem]`、`chunks` 字段、输出文件名是
  兼容性边界；新增字段可以，删除/改义需要升 minor。
- 上游 prompt 串是**外部契约**；改 prompt 会同时改变输出格式，需补后处理
  和测试。
- 0.1.0 起，`/v1/parse` 对不支持的 options 显式 400，而不是静默忽略。
- 本包与 `docmore_paddleocr_vl` 共享无前缀环境变量（`OPENAI_*`、
  `USE_LAYOUT_DETECTION` 等）；同一 shell 同时使用两者时注意覆盖。