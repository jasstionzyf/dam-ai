# dam-ai 实施计划（Implementation Plan）

> 项目定位裁决来源：soujpg opensource-multitenant-strategy（开源通用 DAM 底座，souJpg 是行业用例）。
> dam-ai = DAM 官方 AI 推理组件：独立开源项目，与 souJpg 无名称/代码耦合。
> 仓库：github.com/jasstionzyf/dam-ai（public）+ gitea 172.25.45.244:13000/zhaoyufei/dam-ai（镜像）。

## 一、架构裁决（已定）

```
dam-ai (monorepo)
├── tagger/      → vLLM engine：生成式多模态打标（吞吐红利 + 新模型跟进快）
├── embedder/    → transformers engine（ST + open_clip 双 loader：覆盖 Qwen3-VL-Emb / SigLIP2 / CLIP / DINOv3）
├── registry/    → 模型注册表 + 任务模板注册表
├── server/      → FastAPI 统一 API 层（两个 engine 独立进程、独立 GPU 配额）
└── deploy/      → 每模型一个 compose service（模板参照 docker-compose-embedding.yml 模式）
```

裁决依据（来自 soujpg 实测）：
- 打标是生成式任务，vLLM paged attention + continuous batching 有数量级吞吐优势；结构化输出用 vLLM guided decoding（json_schema），不自己写正则解析。
- embedding 模型生态散（ST 格式 / open_clip 格式），vLLM 覆盖不全（DINOv3 不支持、SigLIP2 pooling 不可控、last-token pooling 要自己对齐）；特征抽取吞吐瓶颈在图片不在模型，vLLM 红利吃不到 → transformers 路线。
- 不塞 TEI：TEI 只管文本，多模态图像塔不吃。

## 二、HTTP 接口标准（已定）

### 1. 端点

```
POST /v1/embeddings          # 文本/图片统一，OpenAI 兼容
GET  /v1/models              # 模型自描述：dims / modalities / normalized
POST /v1/tagging             # 批量打标（模板层，业务接口）
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
// 响应：item 级 status + id 原样透传（pipeline 按 imageSN 回写）
{
  "results": [
    {"id": "228123456", "status": "ok", "output": {...}, "task_version": 3},
    {"id": "pair-42", "status": "error", "error": {"message": "image download failed", "code": "upstream_error"}}
  ]
}
```

模板（registry/tasks.d/*.yaml，Git 管理，不进 MongoDB——修掉 yufei_models 耦合）：

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

模板规则：prompt/schema/params 三件套服务端锁死（输出可比性 = pipeline 回写前提）；
task_version 随响应返回写回图片记录（可定位重刷范围）；评测横评固定 task version 只变 model。

### 4. /v1/models

```json
{"object": "list", "data": [{
  "id": "dinov3-vitb16",
  "object": "model",
  "owned_by": "dam-ai",
  "dam_ai": {"modalities": ["image"], "dims": 768, "normalized": true, "engine": "transformers", "loader": "open_clip"}
}, ...]}
```

## 三、实施阶段

### Phase 0 — 骨架（本仓库初始化）
- [x] GitHub + Gitea 仓库创建
- [x] README（定位 + API 表）
- [ ] LICENSE（AGPL-3.0）
- [ ] 目录骨架：tagger/ embedder/ registry/ server/ deploy/ docs/
- [ ] .gitignore / pyproject.toml / CI（lint+test）

### Phase 1 — embedder MVP（transformers engine）
- [ ] model registry：name → loader(ST|open_clip) / dims / modalities / 预处理(resolution, mean-std, pooling)
- [ ] 首批 4 模型注册：qwen3-vl-embedding-2b(ST) / siglip2-so400m(open_clip) / clip-vit-l14(open_clip) / dinov3-vitb16(open_clip)
- [ ] /v1/embeddings：文本 100% OpenAI 兼容 + image_url content part 扩展 + float16 选项
- [ ] /v1/models / healthz / readyz
- [ ] 验收：openai SDK 纯文本调用通过；四模型 cosine(self)=1.0 自检（对齐 soujpg 评测判据）
- [ ] deploy/compose：单模型一 service（模板 docker-compose-embedding.yml）

### Phase 2 — tagger MVP（vLLM engine）
- [ ] vLLM 子进程/子容器封装：OpenAI 协议透传 + 健康检查
- [ ] 任务模板注册表（YAML + Jinja2 + json schema 校验）
- [ ] /v1/tagging：批量扇出 → vLLM continuous batching、item 级 status、id 透传、失败重试
- [ ] 首批模板：image_caption_metadata / nsfw_check（prompt+schema 从 soujpg 打标链路提炼，去业务耦合）
- [ ] 验收：单图/多图/批量混合失败场景；task_version 回带

### Phase 3 — 开源化打磨
- [ ] docker compose 一键起（无 GPU 退化模式：API 返回 model_unavailable 而非崩溃）
- [ ] 文档：模型接入指南（新增模型 = registry 加一条 + 预处理锁死）、模板编写指南
- [ ] CI：单测（registry 校验 / API schema）+ 集成测试（小模型 CPU smoke）
- [ ] 发布 v0.1.0

## 四、明确不做（边界）

- 不做编排/队列/定时（DAM 主项目管线负责，dam-ai 是无状态推理服务）
- 不做向量索引/检索（ES/OpenSearch 侧职责，embedder 只出向量）
- 不做计费/多租户（enterprise 层，独立私有仓）
- 不做模型训练/微调（只推理）

## 五、与 soujpg 的关系（单向依赖）

soujpg/行业用例 → 调用 dam-ai HTTP API（embeddings + tagging）。
dam-ai 不 import 不配置任何 soujpg 内网资源（MongoDB vcg.yufei_models、gpu0.dev.yufei.com
等耦合全部不带入）。soujpg 现有三个评测容器（8084/8085/8086）是 dam-ai embedder 的
原型验证，迁移即替换。
