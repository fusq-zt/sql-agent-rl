# 运行指南

从仓库根目录执行以下命令。代码公开版对入口和路径做了整理，核心 Agent 与奖励逻辑来自原项目；本次整理只验证了 CPU 功能和入口，没有重新跑 GPU 训练或重测 README 成绩。

## 1. 先阅读和检查

仅需要 Python 3.12+，无需下载数据或模型：

```bash
python -m sql_agent_rl.train --dry-run
python -m unittest discover -s tests -v
```

`train` 默认只打印配置，只有显式传入 `--execute` 才会启动训练。

## 2. 准备训练环境

实验环境为 Linux、Python 3.12、CUDA 12.8、两张 RTX 4090 24GB、约 120GB RAM。为数据、基础模型、FSDP checkpoint 和合并权重预留至少 150GiB 可用磁盘；实际需要取决于保存策略。Windows 可做源码阅读与 CPU 检查，GPU 训练使用 Linux。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-training.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation

python scripts/setup_upstream.py
python -m pip install -e vendor/agent-lightning -c requirements-training.txt
python -m pip install -e .
```

`requirements-training.txt` 固定主要运行依赖，不是跨平台通用环境锁。FlashAttention 构建需要匹配的 CUDA toolkit。不要用新版 Agent Lightning 主分支替换此配置；本项目依赖其 0.3.1 接口和 `runtime/upstream.json` 中的固定提交。

`setup_upstream.py` 只准备该提交并应用 `runtime/agent-lightning.patch`，不安装依赖或启动进程。训练入口会检查框架版本及必要源文件，拒绝混用不同运行实现。vLLM 在这里是 verl 内部采样依赖，不需要额外启动部署服务来训练。

## 3. 准备 Spider 与基座模型

从 [Spider 官方页面](https://yale-lily.github.io/spider) 了解数据，使用 [Agent Lightning 示例所引用的数据包](https://drive.google.com/file/d/1oi9J1jZP9TyM35L85CL3qeGWl2jqlnL6/view)。本仓库不再分发数据库和标注。

固定包 SHA256：

```text
e6d2efb262d9a8a57cd9ddddd63a9c9522bab71775013319b6aa41ea76a28d7f
```

```bash
python -m sql_agent_rl.prepare_data --archive /path/to/spider-data.zip --output data
```

该命令要求 `data` 是新目录，会检查数据库、执行参考 SQL 并生成 `splits/`、`evaluator/` 和 `manifest.json`。训练时只把不含参考 SQL 的任务行交给 Agent。

下载固定模型快照：

```bash
hf download Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --revision 2e1fd397ee46e1388853d2af2c993145b0f1098a \
  --local-dir models/qwen2.5-coder-1.5b
```

## 4. 启动 GRPO

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m sql_agent_rl.train --execute \
  --data data \
  --model models/qwen2.5-coder-1.5b \
  --run-dir runs/grpo
```

默认使用两 GPU、8 个 runner、每题 4 条 rollout 和完整两轮训练。运行目录必须不存在，避免覆盖已有权重。启动后将配置、数据补齐列表、指标与 checkpoint 保存在该目录；Ray 使用短临时目录并绑定 loopback。

主要产物：

```text
runs/grpo/
├── recipe.json
├── resolved_config.json
├── exposure_padding.json
├── traces/                    # 实际 Agent 交互、奖励、训练指标
└── checkpoints/
    ├── selection.json
    ├── best/                  # 按内部开发集保留
    └── global_step_412/       # 完整训练末尾
```

日志、权重和数据都已在 `.gitignore` 中排除。此公开入口只启动新训练，不隐式恢复旧目录；长任务失败时保留现场，由使用者决定后续处理。

## 5. 导出与单题演示

合并完整训练末尾的 FSDP 权重，目标目录必须是新目录：

```bash
python -m sql_agent_rl.export \
  --checkpoint runs/grpo/checkpoints/global_step_412 \
  --target models/sql-agent-grpo
```

若已有可调用的 OpenAI-compatible 模型端点，可在自建教学数据库上观察 Agent 流程：

```bash
python scripts/create_demo_db.py
export OPENAI_API_BASE=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=local-dummy
python -m sql_agent_rl.demo \
  --database runs/demo.sqlite \
  --question 'List each department and its average employee salary.' \
  --model YOUR_SERVED_MODEL_NAME \
  --tokenizer models/sql-agent-grpo
```

`examples/demo.sql` 是自建的部门/员工示例，供理解节点流转，不是 Spider 评测样本或真实模型效果展示。演示入口打印实际 SQL 候选和停止原因，不包含参考答案。
