# dam-ai 基准事实（所有卡必须遵守，不得自行探测重建）

> 本文件从 ~/.hermes/profiles/soujpg/plans/dam-ai-implementation-kanban-split.md
> 「基准事实」节复制。完整 DAG 与各卡范围/验收见该文件（唯一事实源：docs/design.md）。

- 仓库：/data/projects/dam-ai（main 分支）。双远程推送（代理规则相反，别记反）：
  ```bash
  git -c http.proxy=http://127.0.0.1:11700 push github main   # GitHub
  git -c http.proxy= push gitea main                          # Gitea 内网直连
  ```
- 部署机：gpu0（本机），所有服务 docker compose 容器化（用户铁律，不裸跑）。
- 端口分配：embedder=8090 / tagger(vLLM)=8091 / classic=8092。
- 模型源目录（gpu7/gpu0 同路径）：/data1/mlib_data/zhaoyufei_cache/soujpg/models/
  （bge/clip/qwen 系全在此；qwen3-vl-embedding-2b 确认存在）。开跑先 ls 逐项确认，缺的 comment 报告，不要自行下载。
- tools 移植源码：/data/projects/model-infer-api（本地副本），颜色实现在 mcsearch/（colorModel.py ColorModelV2、imageColorPalette.py ImageColorPalette）。
- gpu0 显存现状：3090 已有 vLLM VL 服务（:30000，util 0.60）+ qwen3-vl-embedding 容器（:8084）；gpu7:4080 整卡空闲（vllm-qwen-vl 已停用，FP8 幻觉只影响精度不影响功能冒烟）。
- ifa-test 容器：API 端口 8006（宿主 8206）；python 用 /data/apps/miniconda3/envs/sj/bin/python3.11；
  改代码后必须 kill + nohup bash /tmp/restart_api.sh 重启 uvicorn（--reload 不生效）；
  git pull 后 baseConf 的 fetchOriginalImageUrlUseProxy 会覆盖回 True，须改回 False；
  push 用 git -c http.proxy= push origin yufei-dev。
- 通用规则（写进每卡）：验收不过自省重跑最多 2 轮，仍败即 kanban_block(kind=needs_input) 附根因+已尝试+原始报错；
  长任务心跳 note 带进度数字（N/M）；先小批量 dry-run 再全量；原始命令+输出落 repo 内 reports/
  （/data/projects/dam-ai/reports/，gitignore）并进 kanban_complete(metadata.acceptance)；
  每卡完成 commit + 双远程推送；教训写回 dam-ai skill。
