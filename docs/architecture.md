# 方案与代码阅读

## 1. 从问题到最终 SQL

输入包含用户问题、数据库 ID 和数据库相对路径。`SQLAgent` 从真实数据库读取 Schema，依次调用生成、执行、检查节点；需要修正时进入改写节点，再次执行与检查。

| 节点 | 输入 | 输出 | 是否直接参与策略训练 |
|---|---|---|---|
| `write_query` | 问题、Schema | 首条 SQL | 是 |
| `execute_query` | 当前 SQL | 执行状态与结果预览 | 否，工具操作 |
| `check_query` | 问题、Schema、SQL、执行观察 | 检查反馈 | 否 |
| `rewrite_query` | 问题、Schema、上一条 SQL、执行观察、检查反馈 | 新 SQL | 是 |

`agent.py` 保留三个提示词模板和完整状态图。停止判断使用检查回复中的约定标记，并受最多 3 个 SQL 候选的预算约束；它不是额外训练的停止分类器。改写未解析出 SQL 时沿用上一条候选，不把未执行的回复当成新 SQL。

执行层只允许读取 SQLite 数据。`sql_safety.py` 负责只读连接、权限检查、VM 超时和结果预览；`schema_database.py` 读取表结构。展示给模型的结果可截断，终端判分比较完整查询结果。

当前配置沿用这次运行的 JSON 执行观察格式。它是输入格式设置，不单独作为效果提升的来源；`feedback.py` 保留原始文本格式以便阅读实现。

## 2. 终端奖励与动作选择

参考 SQL 放在 `data/evaluator/gold.json`。任务、模型提示词和工具反馈不包含它；轨迹结束后，评判器才使用参考 SQL：

```text
R = 1，最终 SQL 与参考 SQL 的完整执行结果满足比较规则
R = 0，否则
```

沿用上游比较器的归一化、列排列与空结果规则，不把“SQL 能执行”当成“答案正确”，也不额外训练奖励模型。

Agent Lightning 从真实模型调用中提取 prompt/response token ID。适配器只匹配：

```python
agent_match = r"^(write_query|rewrite_query)$"
```

生成和改写共享模型参数，两类动作都参与优化。检查节点用于决定后续交互，但其生成 token 不进入这里选择的策略损失。

## 3. GRPO 如何连接多轮交互

每个题次采样 4 条 rollout。每条轨迹的终端奖励分配给其中被选中的生成/改写动作，再按同一题次 ID 分组计算相对优势：

```text
A_j = (R_j - mean(R_group)) / (std(R_group) + epsilon)
```

这里的组是在动作展开之后形成的：一条轨迹可能包含一次生成和多次改写，因此动作组大小不固定为 4，各动作的完整 prompt 也可能不同。终端奖励分配不是对中间 SQL 独立正确性的标注。

策略更新使用 verl 的 GRPO/PPO 裁剪实现。当前设置为非对称裁剪范围 `[0.8, 1.3]`、负优势 dual clip 系数 `3`；KL 奖励、KL loss 和熵正则关闭。优势标准化使用样本标准差；单元素组沿用框架特例。项目接入并配置这些能力，没有重新发明 GRPO 损失。

## 4. 数据与训练组织

`prepare_data.py` 从固定 Spider 包中的原始 JSON 建立稳定题目 ID。先检查数据库与参考 SQL，再按数据库划出训练集和内部开发集。三个参考 SQL 执行失败的训练样本不进入训练，官方 dev 的 1,034 题完整保留。

训练集 6,563 题按批大小 32 做确定性补齐，每 epoch 实际暴露 6,592 个题次；训练两轮，共 412 个任务批。这些补齐题次不是新增独立样本。

`train.py` 负责路径配置、数据检查、随机状态与 Ray 生命周期；`callbacks.py` 委托框架执行更新并记录指标，保留内部开发集选择的 best 和最后 checkpoint。README 展示的是完整训练末尾 checkpoint 的观测结果。

训练任务批数不等于原生优化器步数：每批展开出的动作数可变，框架按完整原生 minibatch 执行更新。`trace_adapter.py` 检查实际执行动作与被选择动作一致，避免把不完整轨迹交给策略更新。

## 5. 建议阅读顺序

1. `agent.py`：三个提示词、`SQLAgent.graph()`、`LitSQLAgent.rollout()`。
2. `evaluation_guard.py`：终端奖励从哪里来。
3. `configs/grpo.json` 与 `train.py`：采样和模型更新如何连接。
4. `trace_adapter.py`：哪些节点实际进入训练。
5. `prepare_data.py`：数据划分以及模型输入与参考答案的隔离。

`runtime/` 仅保存固定框架版本及必要兼容补丁；不需要先通读整个框架才能理解项目。Agent Lightning 的 runner/store/adapter、verl 的 GRPO/FSDP 是复用的基础能力。
