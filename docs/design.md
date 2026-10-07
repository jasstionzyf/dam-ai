# dam-ai 设计规范（Design Spec）

> dam-ai = 开源 DAM 栈的官方 AI 推理组件：独立开源项目，不含任何业务耦合。
> 仓库：github.com/jasstionzyf/dam-ai

## 一、架构裁决（已定）

```
dam-ai (monorepo)
├── tagger/      → vLLM engine：生成式多模态打标（吞吐红利 + 新模型跟进快）
├── embedder/    → transformers engine（ST + open_clip 双 loader：覆盖 Qwen3-VL-Emb / SigLIP2 / CLIP / DINOv3）
├── classic/     → 传统 CV 算法 engine（numpy/skimage/faiss：颜色直方图 / 调色板 / PQ 编码，无模型权重，CPU 可跑）
├── registry/    → 模型注册表 + 任务模板注册表
├── server/      → FastAPI server 代码（同一套代码打进各 engine 服务）
└── deploy/      → 每 engine 一个 compose service（模型需独占 GPU 时可再拆）
```

部署拓扑（已裁决：方案 B，无统一网关层）：
- 每 engine 一个独立 compose service，内置完整 server 代码，各自占用独立端口：
  embedder=8090 / tagger=8091 / classic=8092（classic 无 GPU 也能跑）
- 一个 service 内可注册多个模型（registry 寻址，请求 model 字段区分），不做路由层/服务发现
- GPU 配额隔离 = 容器级隔离；某模型需独占 GPU 时拆独立 service（新端口），上游 LB URL 列表追加即可
- 各 service 只暴露自己 engine 的端点（embedder 无 /v1/tagging，tagger 无 /v1/embeddings）

裁决依据（来自原型实测）：
- 打标是生成式任务，vLLM paged attention + continuous batching 有数量级吞吐优势；结构化输出用 vLLM guided decoding（json_schema），不自己写正则解析。
- embedding 模型生态散（ST 格式 / open_clip 格式），vLLM 覆盖不全（DINOv3 不支持、SigLIP2 pooling 不可控、last-token pooling 要自己对齐）；特征抽取吞吐瓶颈在图片不在模型，vLLM 红利吃不到 → transformers 路线。
- 传统 CV 算法（颜色直方图/调色板/PQ 编码，numpy/skimage/faiss）不是深度模型、无权重文件，单列 classic engine（CPU 可跑，支撑无 GPU 退化模式）。
- 不塞 TEI：TEI 只管文本，多模态图像塔不吃。

## 二、HTTP 接口标准（已定）

### 1. 端点

```
POST /v1/embeddings          # 文本/图片统一，OpenAI 兼容
GET  /v1/models              # 模型自描述：dims / modalities / normalized
POST /v1/tagging             # 批量打标（模板层）
POST /v1/classify            # 传统视觉模型（image + task → scores）
POST /v1/chat/completions    # vLLM 透传（逃生口）
GET  /healthz   GET /readyz
```

### 2. /v1/embeddings（OpenAI 兼容 + 最小多模态扩展）

```json
// 请求
{
  "model": "qwen3-vl-embedding-2b",
  "input": [
    "纯文本",
    {"type": "image_url", "image_url": {"url": "https://... 或 data:image/...;base64,..."}}
  ],
  "encoding_format": "float"          // float | float16（海量向量落库省一半）
}
// 响应：object/model/data[{object,index,embedding}]/usage 全按 OpenAI；
// 非标准字段（modality 等）只加不改；维度/模态声明放 /v1/models metadata。
// 纯视觉模型（DINOv3）传文本 → 4xx，错误体 {error:{message,type,code}} OpenAI 风格。
```

规则：纯文本请求逐字段与 OpenAI 一致（openai SDK / LangChain 零改动）；多模态用
content-parts 风格扩展；不学 TEI 把 truncate/compress 等私有参数塞进标准端点。

### 3. /v1/tagging（模板层 = 核心资产）

```json
// 请求：三种姿势（模板 / 模板+参数注入 / inline 自定义）
{
  "task": "stock_photo_metadata",      // 或省略 task 用 inline prompt+schema（实验通道）
  "task_params": {"style": "illustration"},
  "inputs": [
    {"id": "228123456", "image_url": "..."},                 // 单图简写
    {"id": "pair-42", "images": ["https://...a", "https://...b"]}  // 多图（系列判定/AB抽查）
  ],
  "model": "qwen3.5-vl-4b",            // 可选，默认用模板 model_default
  "concurrency": 8
}
// inputs 服务端硬上限 64 条/请求，超出 4xx；更大批量由调用方自行分片（dam-ai 不做队列）
// 响应：item 级 status + id 原样透传（调用方按自有 id 回写）
{
  "results": [
    {"id": "228123456", "status": "ok", "output": {...}, "task_version": 3},
    {"id": "pair-42", "status": "error", "error": {"message": "image download failed", "code": "upstream_error"}}
  ]
}
```

模板（registry/tasks.d/*.yaml，Git 管理，不进数据库——模板与业务库完全解耦）：

```yaml
name: stock_photo_metadata
version: 3
model_default: qwen3.5-vl-4b
allowed_models: [qwen3.5-vl-4b, glm-4.6v]   # 模板可限定模型白名单（换模型要验证）
images: {min: 1, max: 8}                     # 多图任务上限 + token 预算双限制
prompt: |                                    # Jinja2，支持 task_params 注入
  ...
schema: {type: object, properties: {...}, required: [...]}
params: {temperature: 0.2, max_tokens: 1024} # 锁死，调用方不可覆盖
```

模板规则：prompt/schema/params 三件套服务端锁死（输出可比性 = 结果可回写的前提）；
task_version 随响应返回写回资产记录（可定位重刷范围）；评测横评固定 task version 只变 model。

模板分层（已裁决：外部目录合并加载，不开 fork、不进数据库）：
- 内置模板 `registry/tasks.d/*.yaml` 只放通用模板；业务/私有模板放外部目录，不进开源仓
- 外部目录由 env `DAMAI_TASK_DIRS` 指定（冒号分隔可多个），compose volume 挂载进 tagger service；
  同名模板外部覆盖内置（启动时 log 记录覆盖清单）
- 外部模板与内置走同一套 schema 校验，校验失败拒绝启动；allowed_models / params 锁死 /
  task_version 规则对外部模板同等生效，只是存放位置不同
- docs/ 附 example template + 外部目录接入说明（下游接入指南）

### 4. /v1/classify（传统视觉模型）

```json
// 请求（批量，风格与 /v1/tagging 对齐：item 级 status + id 透传；inputs 上限同 64）
{
  "task": "color",
  "model": "color-v1",
  "inputs": [{"id": "228123456", "image_url": "https://..."}]
}
// 响应：输出字段由 task schema 声明（如 hexColors / opqCode / qualityScore）
{
  "results": [
    {"id": "228123456", "status": "ok", "task": "color", "output": {"hexColors": ["#1a2b3c", "..."]}}
  ]
}
```

### 5. /v1/models

```json
{"object": "list", "data": [{
  "id": "dinov3-vitb16",
  "object": "model",
  "owned_by": "dam-ai",
  "dam_ai": {"modalities": ["image"], "dims": 768, "normalized": true, "engine": "transformers", "loader": "open_clip"}
}, ...]}
```

## 三、实施阶段（Roadmap）

### Phase 0 — 骨架
- 仓库初始化
- README（定位 + API 表）
- LICENSE（AGPL-3.0）
- 目录骨架：tagger/ embedder/ classic/ registry/ server/ deploy/ docs/
- .gitignore / pyproject.toml / CI（lint+test）

### Phase 1 — embedder MVP（transformers engine）
- model registry：name → path(/models 本地目录) / loader(ST|open_clip) / dims / modalities / 预处理(resolution, mean-std, pooling)
- 离线加载规范落地：HF_HUB_OFFLINE=1 + TRANSFORMERS_OFFLINE=1 + compose 挂载 /models:ro（见「模型离线加载规范」节）
- 首批 4 模型注册：qwen3-vl-embedding-2b(ST) / siglip2-so400m(open_clip) / clip-vit-l14(open_clip) / dinov3-vitb16(open_clip)
- /v1/embeddings：文本 100% OpenAI 兼容 + image_url content part 扩展 + float16 选项
- /v1/models / healthz / readyz
- deploy/compose：embedder 单 service（端口 8090，内置 4 模型注册，model 字段区分）
- 验收：openai SDK 纯文本调用通过；四模型 cosine(self)=1.0 自检

### Phase 2 — tagger MVP（vLLM engine）
- vLLM 独立 service（端口 8091）：OpenAI 协议透传 + 健康检查
- 任务模板注册表（YAML + Jinja2 + json schema 校验）+ 外部模板目录合并加载（DAMAI_TASK_DIRS，见「模板分层」）
- /v1/tagging：批量扇出 → vLLM continuous batching、item 级 status、id 透传、失败重试
- 首批模板：image_caption_metadata / nsfw_check / translate（通用打标需求，无业务耦合）
- 验收：单图/多图/批量混合失败场景；task_version 回带

### Phase 3 — classic engine（传统 CV 算法引擎）
- `classic/` engine：**纯算法、无模型权重、CPU 可跑**——从 tools 的 mcsearch/ 移植
  （colorModel.py ColorModelV2：HSV 调色板 + LAB 空间平滑直方图；imageColorPalette.py
  ImageColorPalette：faiss.Kmeans 像素聚类；PQ 码本编码），统一接口 `/v1/classify`
  （image + task → codes/scores）；PQ 码本文件随 registry path 寻址，算法常驻内存。
  **码本裁决：直接复用 tools 既有码本文件，禁止重新训练**——ES 存量 opqCode 与新服务
  必须同一码空间，否则颜色搜索直接错乱
- 首批注册（均为通用 DAM 需求，NSFW/质量/people 已由 VL 路径覆盖不迁）：

| 能力 | 输出字段 | 说明 |
|---|---|---|
| 主色提取 + 颜色 PQ | features / opqCode | 颜色搜索基础数据（含 negativeSpace） |
| 调色板提取 | hexColors | 与上者合并为一个 color 服务 |

- 颜色能力无 GPU 依赖，正好支撑「无 GPU 退化模式」：DAM 用户 CPU 即可用颜色搜索
- 偏业务 + 隐私合规敏感的能力（人脸特征、人体部位检测）不纳入，随 tools 淘汰
- 验收：输出字段与既有实现逐项对齐（features/hexColors/opqCode 逐字节一致，
  numpy/skimage/faiss 版本锁定与 tools 现役一致）；
  颜色搜索链路（qColorsInfo 拼色图 → opqCode → ES colorCodes top10）切换前后一致

### Phase 4 — 开源化打磨
- docker compose 一键起（无 GPU 退化模式：API 返回 model_unavailable 而非崩溃）
- 文档：模型接入指南（新增模型 = registry 加一条 + 预处理锁死）、模板编写指南
- CI：单测（registry 校验 / API schema）+ 集成测试（小模型 CPU smoke）
- 发布 v0.1.0

### Phase 5 — souJpg 接入整合（已裁决：走 serviceName2UrlInfo LB，不平移 HttpModelInfer）

souJpg 侧已有现成基建，dam-ai 接入零新增路由代码：

1. **LB 复用**：`souJpg/comm/loadBalancer.py`（@singleton，线程级 least-used +
   FAILURE_COOLDOWN 30s + timeout 120s + retry 2），现役调用方 vlUnifiedFieldMapper /
   bizTagLabeler（qwenVL-chat 已在跑）。baseConf 增加三个 serviceName：

   ```yaml
   serviceName2UrlInfo:
     damai-embed:    ["http://gpu0.dev.yufei.com:8090/v1"]   # embedder service
     damai-tag:      ["http://gpu0.dev.yufei.com:8091/v1"]   # tagger (vLLM) service
     damai-classify: ["http://gpu0.dev.yufei.com:8092/v1"]   # classic service（CPU）
   ```

   多机房 = URL 列表多列各机房实例；DamAiClient 照 vlUnified 的 `self._lb.call(...)`
   模式实现（embeddings/tagging/classify 三方法 + OpenAI SDK 反序列化）。
2. **HttpModelInfer 不平移、不改造**：其能力拆解——URL 池/least-used/失败排除重试
   → LB 全覆盖；modelId+modelInferUrl 表（MongoDB）寻址 → dam-ai registry model 名 +
   baseConf 配置取代；userLevel 分级 → 暂无需求不带入。Phase 4.x 清理时整体删除。
3. **region 容错：死特性不平移**。HttpModelInfer 的 requestRegion 过滤
   （:134-146）只有 nsfwFieldMapper 传过且 URL 池无 region 字段实例，从未生效。
   现网 gpu0/gpu7 同内网无多机房现实；真到多机房时在 LB URL 条目加 region 权重
   （软降级全池，优于 HttpModelInfer 的硬过滤拒绝服务）。
4. **切换面**（gcf 配置即可切换，不动 mapper 内部逻辑）：
   - vlUnifiedFieldMapper：qwenVL-chat → damai-tag（souJpg 业务模板转 yaml 放 image-front-api
     仓 deploy 目录，经 DAMAI_TASK_DIRS 挂载进 tagger，**不进 dam-ai 开源仓**；输出与现行对照后再切）
   - ImageEmbeddingFieldMapper / 语义搜索 CLIP(57490277) → damai-embed
   - QwenVLClient（translate/caption/describe/embed）→ DamAiClient
   - colorFieldMapper/colorPaletteFieldMapper(61624780/39559380) → damai-classify
   - tokenizer(8207) 不动（非 dam-ai 范畴）
5. 验收：LB 三 serviceName 冒烟；单 URL 故障注入 → cooldown 生效切到备用；
   全链路回归（语义搜索/以图搜图/userUpload 管线/颜色搜索）通过后下线 tools 容器

## 四、明确不做（边界）

- 不做编排/队列/定时（DAM 主项目管线负责，dam-ai 是无状态推理服务）
- 不做向量索引/检索（ES/OpenSearch 侧职责，embedder 只出向量）
- 不做计费/多租户（enterprise 层，独立私有仓）
- 不做模型训练/微调（只推理）

## 五、模型离线加载规范（已裁决：档 1 裸目录方案）

私有网络不可访问 HuggingFace Hub 时的标准做法——**挂载本地模型目录 + registry 寻址 +
硬禁 hub 访问**，不引入自建镜像层：

1. **裸目录格式**：每个模型一个普通目录（config.json + model.safetensors + tokenizer 等
   完整文件），提前在有网环境下载/转换好。兼容 ST 格式（modules.json）与 open_clip 格式。
2. **registry 是唯一寻址入口**：API `model` 名 → `path`（容器内约定挂载点 `/models`）
   只在 registry/models.yaml 一处映射；代码只用 `from_pretrained(local_path)`。
3. **compose 挂载 + 离线环境变量（必配）**：

   ```yaml
   services:
     embedder:
       volumes: ["/data1/models:/models:ro"]
       environment:
         - HF_HUB_OFFLINE=1       # 禁一切 hub 访问，离线加载不超时重试
         - TRANSFORMERS_OFFLINE=1
   ```

4. HF cache 快照格式（repo-id 引用）天然兼容（HF_HUB_OFFLINE 下走 snapshots 解析），
   不需要额外代码；自建 hub 镜像（HF_ENDPOINT/ModelScope）不采用，文档提及即可。
5. 模型目录是普通文件：可 rsync/tar 归档（归档路径写回 registry 注释）。

## 六、依赖边界

dam-ai 是独立开源项目，不 import、不配置任何下游业务资源（业务库、内网地址、私有模型仓
全部不带入）。下游应用通过 HTTP API 单向调用（embeddings / tagging / classify），
dam-ai 不感知调用方。
