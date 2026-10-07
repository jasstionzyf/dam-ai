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
- [ ] model registry：name → path(/models 本地目录) / loader(ST|open_clip) / dims / modalities / 预处理(resolution, mean-std, pooling)
- [ ] 离线加载规范落地：HF_HUB_OFFLINE=1 + TRANSFORMERS_OFFLINE=1 + compose 挂载 /models:ro（见「模型离线加载规范」节）
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

### Phase 4 — 接入 souJpg，替换 tools 容器（model-infer-api gpu7:1020）旧接口

目标：dam-ai embedder/tagger 上线后，souJpg 后端（ifa-test → 生产）推理调用全部切到
dam-ai，下线 tools 容器 `/rest/model-infer`（modelId 寻址）路径，回归全绿。

现状依赖（2026-10-08 盘点，迁移面）：
- `gcf.modelInferAPI`（baseConf_test.yaml `modelInferAPI: http://…:1020/rest/model-infer`）
  → `HttpModelInfer.inferWithParams`，modelId 字符串寻址：
  - 57490277 CLIP（图/文 embedding）：searchInfoBuilder.setTextEmbedding（语义搜索，强制
    semanticSearchMode=1）、opNodeExecuteEngine（以图搜图）、comm.py 离线批量任务
  - 34593113 SeamlessM4T（translate）：searchInfoBuilder.translateInner / keywordsHelper /
    Text2ImageOpNode —— **大部分已被 QwenVLClient.translate（vLLM :30000）替代，仅残路径**
  - 55774328 text embedding（semanticSearchMode=0，当前被强制 1，死路径）
  - 99183490 BLIP / 26311198 VL describe：已被 QwenVLClient（qwenVL.chatApi）替代
  - opNodeExecuteEngine 其余 modelId（SD/codeFormer/faceSwap/realESR/nsfw 9877205/4483468/
    61624780/49305217/29311198 等）：图像工具类，多数已停用（tools 容器 Exited 5 天）
- 现役推理端点（不走 1020）：vLLM :30000（chat）、:30001（Qwen3-VL-Emb embeddings）、
  tokenizer :8207 —— 这些也统一切到 dam-ai 单入口

步骤：
- [ ] 4.1 dam-ai embedder 注册 clip-vit-l14（open_clip loader，与 57490277 同权重/同预处理，
      输出向量必须与现网 CLIP 向量同空间——ES 1.07 亿 imageEmbedding 不重建，只换推理提供方）
- [ ] 4.2 dam-ai tagger 注册 translate / caption / describe 任务模板（对齐 QwenVLClient
      现有 prompt，输出逐字节兼容）
- [ ] 4.3 ifa-test：baseConf 增加 damAi 端点配置；新增 DamAiClient（embeddings + tagging +
      chat 透传），QwenVLClient 与 HttpModelInfer(CLIP/translate) 调用点改为 DamAiClient
- [ ] 4.4 对照验收：同输入下新旧接口向量 cos ≥ 0.999（CLIP 路径必须 =1.0，同权重）、
      translate/caption 输出一致；ES 语义搜索返回 top10 与切换前一致
- [ ] 4.5 全量回归：soujpg E2E（soujpg-e2e-test）+ 前端浏览器 E2E（soujpg-web-e2e-test），
      重点：语义搜索、以图搜图、userUpload 管线（caption/title/oKws）、AI 编辑、翻译链路
- [ ] 4.6 生产（gpu7 compose）同步切换 + 观察，下线 tools 容器（docker rm，镜像保留）
- [ ] 4.7 移除 gcf.modelInferAPI / HttpModelInfer 死代码路径（55774328、99183490、26311198、
      opNode 停用 modelId），baseConf 清理
- [ ] 回归标准：全部既有回归用例通过；tools 容器下线后 48h 无 1020 端口调用（日志零命中）


### 第三类模型裁决：classic engine（传统视觉分类器/检测器，Phase 4 前置）

tools 容器里除 CLIP embedding（embedder 覆盖）和 VL 生成（tagger 覆盖）外，还有第三类
「传统视觉模型」，vLLM/ST/open_clip 三个 loader 都不吃（TF-Keras/ONNX/detectron 格式）。
souJpg 在用且不能断的（userUpload 管线 mapper 顺序号）：

| modelId | 功能 | 调用点 | 裁决 |
|---|---|---|---|
| 61624780 | 主色提取 + color PQ | colorFieldMapper(11)、**qColorsInfo 颜色搜索**（拼色图→opqCode→ES） | ✅ 进 dam-ai（DAM 通用需求） |
| 39559380 | 调色板 hexColors | colorPaletteFieldMapper(12) | ✅ 进 dam-ai（同上，颜色类合并一个 color 服务） |
| 28717010 | NSFW 打分 | nsfwFieldMapper(0)，nsfw_score | ✅ 进 dam-ai（DAM 通用需求；现权 = 本地 TF Keras nsfw 家族，/data1/.../models/nsfw/） |
| 32484322 | 图像质量分 | imageQualityFieldMapper(13) | ✅ 进 dam-ai（DAM 通用需求） |
| 49931946 | 人体部位检测 | bodyPartFieldMapper | ⚠️ 暂不迁：偏业务 + 隐私合规敏感，留 tools 或随 tools 一并淘汰（另观察） |
| 49305217 | 人脸特征 + OPQ | faceOpqCodesMapper / peopleFieldsMapper | ⚠️ 同上暂不迁 |

- [ ] dam-ai 增加 `classic/` engine：按模型格式配 loader（TF-Keras saved_model / ONNX /
      ultralytics 等），统一接口 `/v1/classify`（image + task → scores），registry 同样
      path 寻址 + 离线加载
- [ ] 首批 classic 注册：color(61624780 含 PQ)、palette(39559380)、nsfw(28717010)、
      quality(32484322)——从 tools 容器提取权重与后处理，输出字段对齐现有
      （hexColors/opqCode/nsfw_score/qualityScore），对照验收新旧一致
- [ ] 颜色搜索专项回归：qColorsInfo → 拼色图 → opqCode → ES colorCodes top10 切换前后一致
- [ ] 人脸/人体部位：peopleNum 字段当前由 VL（vlUnifiedFieldMapper）+ BodyPart 双路出，
      确认 VL 路径全覆盖后可淘汰 bodyPart/faceOpq 依赖（独立验收项）

## 四、明确不做（边界）

- 不做编排/队列/定时（DAM 主项目管线负责，dam-ai 是无状态推理服务）
- 不做向量索引/检索（ES/OpenSearch 侧职责，embedder 只出向量）
- 不做计费/多租户（enterprise 层，独立私有仓）
- 不做模型训练/微调（只推理）

## 四·五、模型离线加载规范（已裁决：档 1 裸目录方案）

私有网络不可访问 HuggingFace Hub 时的标准做法——**挂载本地模型目录 + registry 寻址 +
硬禁 hub 访问**，不引入自建镜像层：

1. **裸目录格式**：每个模型一个普通目录（config.json + model.safetensors + tokenizer 等
   完整文件），提前在有网环境下载/转换好。兼容 ST 格式（modules.json）与 open_clip 格式，
   即现 soujpg /data1/.../models/ 的组织方式，可直接平移。
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
5. 模型目录是普通文件：可 rsync/tar 归档（归档盘路径写回 registry 注释）。

## 五、与 soujpg 的关系（单向依赖）

soujpg/行业用例 → 调用 dam-ai HTTP API（embeddings + tagging）。
dam-ai 不 import 不配置任何 soujpg 内网资源（MongoDB vcg.yufei_models、gpu0.dev.yufei.com
等耦合全部不带入）。soujpg 现有三个评测容器（8084/8085/8086）是 dam-ai embedder 的
原型验证，迁移即替换。
