# 架构文档

> 适用版本：0.1.0 · 最后更新：2026-11-26
>
> 本文面向维护者，回答三个问题：系统边界在哪里、一页文档如何流过两个模型、
> 哪些地方被刻意做成了可替换的扩展点以及为什么。接口细节见
> [契约文档](contracts.md)，安装/构建/发布见 [说明文档](guide.md)，
> 实测记录与踩坑见 [验证与记忆文档](verification.md)。

## 1. 系统边界

`docmore-paddleocr-vl-16` 是一个**文档转换编排层**，本身不训练、不托管任何模型：

- 第一阶段（版面检测）在**本进程内**通过 ONNX Runtime 运行
  PP-DocLayoutV3；
- 第二阶段（区域识别）把裁剪图通过 HTTP 发给一个**外部**
  OpenAI 兼容推理服务（PaddleOCR-VL 1.6，通常由 llama.cpp 提供）。

因此正确运行需要两样外部条件：`inference.onnx` 版面模型文件，以及一个
可达的 PaddleOCR-VL 1.6 端点。

| 在边界内 | 在边界外 |
| --- | --- |
| PDF 栅格化 / 图片解码（委托 `docmore-utility`） | PaddleOCR-VL 1.6 权重加载与推理服务 |
| PP-DocLayoutV3 ONNX 推理与后处理 | 版面模型的训练、导出、版本发布 |
| 区域裁剪、任务提示词路由、并发调度 | OCR 结果的业务语义校验（发票号、身份证号等） |
| OTSL→HTML、公式归一化、Markdown/JSON 组装 | PDF 文本层抽取、原生 PDF 结构解析 |
| 可选 DoclingDocument 导出 | Docling 生态的后续 chunk / embedding |
| 可选 FastAPI/OpenAI 兼容协议层 | 任务队列、持久化、多租户计费 |

### 1.1 部署形态

1. **库 / CLI**：本机装 `[layout]`，直接读文件、写目录；上游端点可在本机或远端。
2. **HTTP 服务**：`[server]` 额外装 FastAPI；服务进程仍然只做编排，识别推理在
   上游端点完成，所以服务端不要求 GPU。
3. **容器**：镜像不打包 124 MB 的 ONNX 文件，运行时挂载并设置
   `LAYOUT_MODEL_DIR`。
4. **无版面模式**：不装 `[layout]` 或 `USE_LAYOUT_DETECTION=false` 时，整页作为
   一个 `ocr` 区域送 `OCR:`，功能降级但不阻断。

### 1.2 两个容易混淆的端口

| 端口 | 归属 | 默认值 | 配置变量 |
| --- | --- | --- | --- |
| 上游 PaddleOCR-VL 1.6 推理服务 | 外部依赖 | `8113` | `OPENAI_API_BASE` |
| 本包 HTTP 服务 | 本包 `server` | `18910` | `PADDLEOCR_VL_16_SERVER_PORT` |

`OPENAI_API_BASE` 指向 8113，不是本服务自己的监听端口。

### 1.3 模型别名与流水线模式

服务端对外只提供六个 `paddleocr-vl-1.6-*` 模型别名。`mode.py` 把别名映射为
`PipelineMode`：

| 别名 | 模式 | 版面 | 整页 prompt |
| --- | --- | --- | --- |
| `paddleocr-vl-1.6-doc` | `doc` | 是 | 按块标签派生 |
| `paddleocr-vl-1.6-table` | `table` | 否 | `Table Recognition:` |
| `paddleocr-vl-1.6-text` | `text` | 否 | `OCR:` |
| `paddleocr-vl-1.6-formula` | `formula` | 否 | `Formula Recognition:` |
| `paddleocr-vl-1.6-seal` | `seal` | 否 | `Seal Recognition:` |
| `paddleocr-vl-1.6-chart` | `chart` | 否 | `Chart Recognition:` |

非 `doc` 模式在服务端被翻译成每页的 `BatchInputItem(prompt, prompt_type)`，
因此共享的 manager 仍保持 `doc` 模式，无需为每个别名重建 ONNX 会话。
HTTP 层对六个别名以外的 model 返回 404 `model_not_found`。

## 2. 组件视图

```text
                  ┌──────────────────────────────┐
  PDF / 图片 ────▶ │ input.py                     │
                  │ docmore-utility 栅格化        │
                  └──────────────┬───────────────┘
                                 │ list[PIL.Image]
                                 ▼
                  ┌──────────────────────────────┐
                  │ InferenceManager.generate()  │  ← 稳定契约
                  └───────┬──────────────┬───────┘
                          │              │
             第一阶段      │              │      第二阶段
        ┌─────────────────▼──┐      ┌────▼──────────────────┐
        │ layout.py           │      │ model/client.py        │
        │ PP-DocLayoutV3Onnx  │      │ VLMClient              │
        │ + ONNX Runtime      │      │ OpenAI Chat Completions│
        └─────────┬───────────┘      └────┬───────────────────┘
                  │ LayoutBox[]            │ GenerationResult
                  └──────────┬─────────────┘
                             ▼
                  ┌──────────────────────────────┐
                  │ postprocess.py / table.py    │
                  │ OTSL→HTML、LaTeX、Markdown    │
                  └──────────────┬───────────────┘
                                 │ chunks / markdown / images
             ┌───────────────────┼────────────────────┐
             ▼                   ▼                    ▼
      api.py               docling_bridge.py     server/parse.py
      md/html/json         DoclingDocument      unified-parse/v1
```

### 2.1 模块职责

| 模块 | 职责 | 不负责 |
| --- | --- | --- |
| `input.py` | PDF/图片 → RGB PIL；页范围；最小边长放大 | 版面、识别 |
| `settings.py` | 全包配置单例；版面模型目录自动解析 | HTTP 服务配置 |
| `layout.py` | ONNX 推理 + NMS/合并/掩码转多边形/unclip/排序 | 区域识别 |
| `geometry.py` | 版面框重叠过滤（小框、公式、图片/表格例外） | 模型推理 |
| `prompts.py` | label → task → prompt 路由与开关策略 | 内容后处理 |
| `mode.py` | 六个模型别名 ↔ 流水线模式；整图单块 prompt/label | 具体识别与后处理 |
| `model/__init__.py` | 两阶段编排、并发、失败降级、chunks 组装 | 文件 I/O |
| `model/client.py` | OpenAI 协议适配、重试、退化重复检测 | 业务语义 |
| `postprocess.py` | 区域输出归一化、块级 Markdown | 阅读顺序 |
| `table.py` | OTSL token → 带 rowspan/colspan 的 HTML | 版面判断 |
| `api.py` | `convert()`、批量文件、输出文件布局 | 推理细节 |
| `cli.py` | Click 参数到 `convert()` 的薄封装 | 业务逻辑 |
| `docling_bridge.py` | chunks → `DoclingDocument` | 修改 chunks |
| `server/` | FastAPI 路由、鉴权、请求限制、响应整形 | 新增识别能力 |

## 3. 数据流

### 3.1 文件到页

`convert()` 先用 `get_supported_files()` 展开文件或目录，然后逐个文件
`load_file()`：

- 内容类型由字节判断（`docmore-utility.identify`，未装 `[detect]` 时回退
  `%PDF-` 魔数），不依赖扩展名；
- PDF 用 `docmore_utility.pdf_to_images()` 渲染：默认 200 DPI、短边至少
  1024 px、`flatten=True` 摊平透明背景、返回 PIL；
- 独立图片短边不足 1536 px 时用 LANCZOS 放大；
- `page_range` 是 **1 起** 的字符串（`"1-5,7"`），先解析成有序去重的页码再
  转成 0 起的索引。

一页对应一个 `BatchInputItem(image=...)`；`convert()` 默认每 28 页组成一个
batch（`batch_size` 可覆盖）。

### 3.2 第一阶段：版面检测

每个 batch 交给 `InferenceManager.generate()`。单页流程：

1. `image.convert("RGB")`，记录 `page_box = [0, 0, W, H]`。
2. 若调用方给了 `BatchInputItem.prompt`，或当前模式不是 `doc`（五个整图
   单块别名），跳过版面检测，整页作为一个区域：label 取
   `prompt_type` / 模式内置 label（默认 `settings.PROMPT_LABEL`，即
   `ocr`），prompt 取显式值或模式内置任务串。这是“强制单任务”模式，也是
   使用 `Spotting:` 的唯一入口。
3. 否则 `_detect_boxes()`：
   - `PPDocLayoutV3Onnx.predict_image()` 把图缩放到 800×800，构造
     `im_shape` / `image` / `scale_factor` 三个输入，运行 ONNX；
   - `postprocess_layout()` 依次做：分数阈值、同类 NMS（同/异类 IoU 阈值
     不同）、超大幅 `image` 框过滤、按类 containment 合并
     （`LAYOUT_MERGE_MODE`）、按模型 order 排序、从分割掩码提取多边形、
     unclip 扩张、裁剪到页面范围、生成 `LayoutBox`；
   - `geometry._filter_overlap_boxes()` 去掉 `reference`、小于 6 px 的框；
     `inline_formula` 与任何框重叠 > 0.5 时丢弃公式框；一般框重叠 > 0.7
     时按面积保留大者；涉及 `image/table/seal/chart` 的重叠做保守处理；
   - `layout.normalize_reading_order()` 只对“上下堆叠、横向重叠 ≥ 45%”的
     纯文本块做相邻交换，并重编号 `order`。
4. 若 ONNX 不可用、模型缺失、推理抛异常或检出 0 个框，回退为整页单个
   `ocr` 框。GPU 推理异常时先在锁内重建 CPU 会话重试一次。

### 3.3 区域路由

路由发生在 `_generate_one()` 的循环里，顺序很重要：

| 条件 | 动作 |
| --- | --- |
| label ∈ `image/header_image/footer_image` | 裁剪为图片资产 + Markdown `![](...)`，不送模型 |
| `header_image/footer_image` 且未开启页眉页脚 | 直接跳过 |
| label = `formula_number` 且 `SHOW_FORMULA_NUMBER=false` | 跳过 |
| `prompt_for_label()` 返回 `None` | 丢弃；若 task 是 seal/chart（识别被关闭），改为图片资产 |
| 其余 | 裁剪后加入识别任务队列 |

`prompt_for_label()` 的规则：

- `header/footer/number/footnote/aside_text/reference` 默认丢弃；
  `INCLUDE_HEADERS_FOOTERS=true` 时按 `OCR:` 识别；
- `USE_SEAL_RECOGNITION=false` / `USE_CHART_RECOGNITION=false` 让对应区域
  不送模型，但保留为图片，避免信息丢失；
- `display_formula` 与 `inline_formula` 共用 `Formula Recognition:`，但后处理
  分别渲染为块级 `$$` 与行内 `$`；
- 未知 label 回退为 `ocr`。

### 3.4 第二阶段：并发识别

- **页级并发**：`generate()` 对多页 batch 使用
  `ThreadPoolExecutor(max_workers=MAX_WORKERS)`，`pool.map` 保持输入顺序。
- **区域级并发**：单页内部对裁剪区域再用一个线程池，worker 数为
  `min(MAX_WORKERS, 区域数)`。
- 服务端用 `PADDLEOCR_VL_16_SERVER_MAX_CONCURRENT` 信号量限制同时在飞的
  请求。`/v1/chat/completions` 与 `/v1/parse` 虽是 `async def`，但用
  `asyncio.to_thread` 把 `generate()` 放到工作线程执行，信号量也在工作线程
  内获取——OCR 期间事件循环保持可响应。
- 版面 ONNX 会话的懒加载用 `threading.Lock()` 做双重检查：并发首请求只会
  构建一个会话；GPU 推理失败时在同一把锁内构建 CPU 回退会话并重试一次。
- 每个任务调用 `VLMClient.complete(crop, prompt)`：图片在前、任务提示词在后，
  不使用 system prompt；`temperature=0.0`、`top_p=1.0` 起始。
- 空输出或退化重复（末尾同一 token 反复）会升温度重试：
  `temperature = min(T + 0.2 × attempt, 0.8)`，重试时 `top_p=0.95`，最多
  `MAX_RETRIES + 1` 次；异常同样重试，最终返回 `GenerationResult(error=True)`。

### 3.5 后处理与组装

`normalize_region_content(label, task, raw)` 按任务归一化：

- `table`：依次尝试“已是 `<table>` → 去 `<otsl>` 包装后 OTSL 解码 →
  Markdown 管道表 → 保留纯文本”；
- `chart`：同上；实际端点返回 Markdown 管道表时会被转成 HTML，只返回说明
  文字时保留为文本；
- `formula` / `inline_formula`：去掉 `$$`、`\[ \]`、`\( \)`、`$` 和
  `equation` 等环境包装，得到裸 LaTeX；
- `ocr` / `spotting` / `seal`：CRLF→LF、去代码围栏、去首尾空白。

随后按最初的 `boxes` 顺序（即阅读顺序）组装：

- 图片区域：`chunks` 追加空 content 的块（带 `asset`），Markdown 追加图片链接；
- 识别成功且非空的区域：追加块，并按 label 渲染 Markdown
  （`doc_title → #`、`paragraph_title → ##`、display formula → `$$`、
  inline formula → `$`、table/chart 表格原样 HTML、图表说明文本段落化、
  其余段落化）；
- 识别失败的区域不产生块；
- `raw` 用 `<label>\ncontent` 形式拼接，便于调试。

页级 `error` 的定义是 `not markdown and failures > 0`：只有当整页没有任何
Markdown（包括图片链接）且确实有识别失败时才为真。

### 3.6 文件输出

`save_merged_output()` 把多页结果写到 `out/<文件名>/`：

```text
<name>.md                阅读顺序 Markdown
<name>.html              当前与 Markdown 内容相同
<name>.json              pages[].blocks[]（id/reading_order/label/bbox/content/asset）
<name>_metadata.json     页数、token、块数、图片数
imgs/page_0001/img_in_<label>_box_<x1>_<y1>_<x2>_<y2>.png
<name>.docling.json/.md  --format docling 时
```

多页输入时图片按 1-based 页目录分桶，同样的 bbox 不再互相覆盖；
`chunks[].asset` 记录图片块对应的资产键，下游（unified-parse、Docling）
按这个键精确取图，不再靠尺寸猜。只有直接调用 `generate()` 且未提供
`page_index` 时才保持旧的 `imgs/<name>.png` 形式；旧格式结果在
`save_merged_output()` 中仍会被去重改写。

HTML 目前是 Markdown 的直接副本；表格块本身是 HTML，因此并非纯 HTML 文档。

### 3.7 HTTP 服务路径

```text
请求图片（base64 / data URI / http(s) URL）
  → 解码或下载为 PIL（URL 下载与 generate() 都在 asyncio.to_thread 中）
  → BatchInputItem[]（携带原始页号与 include_headers_footers 覆盖）
  → manager.generate(batch)        # 与 CLI 完全相同的两阶段
  → build_parse_result()           # unified-parse/v1
```

`/v1/chat/completions` 把每张图当作一页，拼接各页 Markdown 返回，并额外
报告失败页数。服务端只在应用生命周期内缓存一个 manager，布局 ONNX 会话
也随之复用；不缓存识别结果。

URL 下载由 `server/images.py` 负责：只允许 `http/https`，默认拒绝解析到
loopback / 私网 / link-local 的地址（`ALLOW_PRIVATE_URLS=1` 才放行），每次
重定向都重新校验，响应体超过 `MAX_IMAGE_BYTES` 立即中止。`ALLOW_URL_FETCH=0`
可以整体关闭远程取图。base64、data URI 与 URL 共用同一个大小上限和解码入口。

## 4. 失败与降级

| 失败点 | 行为 | 结果 |
| --- | --- | --- |
| 没装 `onnxruntime` / 找不到 `inference.onnx` | `layout_available()` 为 false；`_detect_boxes` 直接回退 | 整页 `OCR:` |
| 版面推理抛异常（如 GPU 缺 cuDNN） | 在锁内重建 CPU 会话并重试一次 | 成功则继续，否则整页回退并打印一次警告 |
| 版面检出 0 框 | 回退整页 | 整页 `OCR:` |
| 某个区域请求失败 | 跳过该块，计入 failures | 其余块正常；必要时页级 `error=True` |
| 空输出 / 退化重复 | 升温度重试 | 最终失败则跳过该块 |
| 模型输出不是合法 OTSL/HTML | 管道表或纯文本兜底 | 信息不丢失 |
| 图表只返回说明文字 | 作为文本输出 | 不丢图表描述 |
| URL 图片下载失败 / 超时 / 超大 | 400 + 错误信封 | 调用方修正来源 |
| URL 指向私网且未开启 `ALLOW_PRIVATE_URLS` | 400（SSRF 防护） | 默认不允许内网探测 |
| `docling-core` 未安装 | 捕获 `ImportError`，打印提示 | 其余输出照常 |
| `convert()` 单文件异常 | 捕获、打印、继续下一个文件 | 批次不中断 |
| 服务端所有页失败 | 502 + OpenAI 错误信封 | 无 200 空结果 |
| 服务端部分页失败 | 200 + `X-PaddleocrVl16-Partial: 1`，parse 结果带 warning | 可用部分结果 |

降级设计的原则是：**任何单点失败都不应让整份文档不可用**，除非所有页都
没有任何可输出内容（服务端会明确报 502）。

## 5. 关键设计决策

| 决策 | 理由 |
| --- | --- |
| 两阶段、两个独立模型 | 版面是小而快的专用 ONNX，可本地 CPU 跑；识别是大 VLM、可远端扩容。两者升级互不绑定。 |
| 按区域匹配任务提示词 | 整页送 VLM 时表格/公式/图表容易混排；裁成区域后按类型提示，输出格式更可控（OTSL/LaTeX）。 |
| `chunks` 作为内部稳定契约 | CLI、Docling、HTTP 三种出口共享同一份结构化结果，避免复制三套解析逻辑。 |
| 版面模型懒加载并缓存 | 服务端多请求复用同一个 ONNX 会话；库模式不用版面时完全不导入 ONNX。 |
| 阅读顺序做局部后处理 | PP-DocLayoutV3 的 order 在表单/证件类页面上会上下颠倒；完全重排风险大，只交换明显的同列堆叠块。 |
| 贪心解码 + 升温度重试 | OCR 需要确定性；只有空输出/退化重复才升温，正常页不受随机性影响。 |
| 关闭识别时保留图片资产 | 印章/图表即使不做文字识别，也不应静默丢掉版面信息。 |
| 图表兼容表格与说明文字 | PaddleOCR-VL 1.6 实际返回 Markdown 管道表或说明文字；两种都要能落到 Markdown/JSON。 |
| `Spotting:` 只作为显式任务 | PP-DocLayoutV3 没有 spotting 类，自动路由会把普通文本误送 spotting；强制 prompt 更诚实。 |
| 可选依赖拆分 extras | 库用户可能只要 Markdown；服务端不需要 `[layout]`（无版面模式）；Docling 用户才装 docling-core。 |
| 服务配置与流水线配置分离 | HTTP 端口/鉴权/限额与模型/版面参数生命周期不同。 |
| 资产键带页前缀 | 文件名只含 label+bbox 时，多页相同 bbox 会覆盖；页目录让资产天然唯一，`chunk.asset` 让出口不再靠尺寸猜图。 |
| URL 取图默认拒绝私网 | 解析服务常部署在内网边缘；需要内网图片的部署显式打开 `ALLOW_PRIVATE_URLS`。 |
| 流水线调用 offload 到线程 | `generate()` 是 CPU/IO 混合的同步代码；在 `async def` 中直接调用会串行化所有请求并拖住 `/health`。 |

## 6. 扩展机制及其存在理由

### 6.1 新增版面 label / 新任务类型

扩展点：`prompts.py` 的 `LABEL_TASK_MAPPING`、`TASK_PROMPTS`，以及
`postprocess.py`、`docling_bridge.py`、`server/parse.py` 中的 label 映射。

为什么是表而不是分支：版面模型的 25 类与 PaddleOCR-VL 的 6 种任务不是
一一对应；映射表让“新增 label → 归属已有任务”成为改数据而不是改流程。
新增任务类型时，需要同时补：prompt 常量、内容归一化、Markdown 渲染、
Docling 映射、unified-parse block 类型。漏掉任何一处都会表现为“块存在但
出口丢内容”。

### 6.2 替换版面模型

扩展点：`PPDocLayoutV3Onnx`（模型目录 + 固定输入名/输出索引）。
`LAYOUT_MODEL_DIR` 可以指向任意符合该契约的目录；`resolve_layout_providers()`
负责把 `auto/directml/cuda/cpu` 解析成 ONNX Runtime EP。

为什么保留 `PPDocLayoutV3Onnx` 而不是抽象接口：当前只有一个后端，过早抽象
没有收益；真正的边界在 `predict_image() -> (list[LayoutBox], ndarray)`。

### 6.3 替换第二阶段的 VLM

扩展点：`VLMClient`。它只依赖 OpenAI Chat Completions 协议，所以 llama.cpp、
vLLM、网关代理都可以作为上游。`normalize_base_url()` 让裸地址和带 `/v1`
的地址都可用。换上游时主要改 `OPENAI_API_BASE`、`OPENAI_MODEL_NAME` 和
`prompts.py` 的任务串。模型别名/整图单块模式集中在 `mode.py`：新增模式
需要同时补 `PipelineMode`、`MODES_BY_NAME` 与 `_BLOCK_TYPE_MAP` 的 label。

### 6.4 新增输出格式

扩展点：`BatchOutputItem.chunks` / `images` / `markdown`。
`api.py`、`docling_bridge.py`、`server/parse.py` 都只消费这些字段。

为什么 chunks 用像素 bbox 而不是归一化坐标：版面模型输出就是像素坐标，
同一页内无需换算；导出层若需要归一化可自行除以 `page_box`。

### 6.5 新增 HTTP 能力

扩展点：`server/router.py` 的 `create_app()` 与 `server/parse.py` 的响应整形。
所有路由共享 `_get_manager()` 和信号量，新路由不会重复创建模型客户端。
图片来源同样是一个扩展点：`server/images.py` 的 `load_image_source()` 负责
base64 与 URL 两种来源，路由只关心“拿到 PIL 图”。

### 6.6 新增配置

扩展点：`Settings`（无前缀，流水线）与 `ServeSettings`
（`PADDLEOCR_VL_16_SERVER_` 前缀，HTTP）。服务端读取流水线配置但从不修改它；
测试只 monkeypatch 单例即可关掉版面。

## 7. 状态与生命周期

- `settings` 是模块级单例，导入时读取环境变量/`.env`/`local.env`。
- `convert()` 默认每次调用创建一个 `InferenceManager`；也可以传入
  `model=` 复用。
- `InferenceManager` 内部惰性创建 ONNX 会话并缓存；`_layout_lock` 保证并发
  首调只建一个会话，`_layout_failed` 和 `_layout_cpu_fallback` 是该 manager
  的进程内状态，一旦降级成功或失败，后续页沿用，不会每页重试。
- 服务端每个 `create_app()` 创建一个 manager，进程内共享；manager 固定为
  `doc` 模式，单块模式通过每请求的 `BatchInputItem.prompt`/`prompt_type`
  注入。请求之间不共享 OCR 结果，只共享模型会话。
- 没有跨页阅读顺序、跨页去重或文档级状态；`chunks` 的 id 由 `api.py`
  按 `p{page}-b{index}` 生成。

## 8. 当前非目标与已知边界

- 不提供异步任务队列；HTTP 请求同步等待整个批次完成。
- 不做跨页阅读顺序（例如跨栏续接、跨页表格）。
- URL 取图只做 DNS 解析级 SSRF 过滤，不做“解析后固定连接到该 IP”，因此
  理论上仍存在 DNS rebinding 的时间窗；默认关闭私网访问已消除主要风险。
- `/v1/parse` 的 `options` 只支持 `pages`、`include_images`、
  `include_bounding_boxes`、`include_headers_footers`；其余值显式拒绝
  （400 `unsupported_option`）。
- HTML 输出是 Markdown 副本，不是独立渲染的 HTML 文档。
- `Spotting:` 没有自动路由；需要显式 `prompt`/`prompt_type`。
- 图表识别只把模型返回的表格或说明文字转成一种表示，不额外保留裁剪图
  （关闭图表识别时仍保留图片资产）。
- 版面阶段是页级/区域级线程池嵌套；极端页数下线程数会放大。