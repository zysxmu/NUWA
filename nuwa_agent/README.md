# NUWA-Agent — 服务器部署包

本文件夹是**可直接上传到 Linux 服务器运行**的干净版本，已剔除：
本地含 API key 的 `config.py`、`.bak` 备份、`__pycache__`、`output/` 运行结果、权重文件（太大，单独传）。

---

## 1. 目录结构

```
nuwa_server_upload/
├── nuwa_agent/                 # 全部源码（run 目录，脚本在此执行）
│   ├── config.py               # 服务器版配置（BASE_DIR 派生路径 + 环境变量读 key）
│   ├── main.py                 # 主入口（交互式：蛋白序列 → 宿主 → 约束）
│   ├── run_demo.py             # 一键 demo（EGFP 片段 + E.coli，自动降参）
│   ├── orchestrator.py         # Phase 1 三层 CoT 选模型 + 约束决策
│   ├── model_registry.py       # NUWA 域模型加载 + 熵引导生成
│   ├── evaluator.py            # 3 软目标 (TE/Stability/Expression) + 6 硬约束
│   ├── constraint_checker.py   # 约束可行性判定
│   ├── pareto_selector.py      # NSGA-II 非支配排序 + 超体积
│   ├── round_deliberation.py   # 专家 Agent 讨论 + Central LLM 决策
│   ├── feedback_analyzer.py    # 分数相关性分析与旧版反馈实现
│   ├── iteration_controller.py # 迭代循环、策略执行 + 收敛判定
│   ├── species_resolver.py     # 物种名 → (domain, class_id)
│   ├── .gitignore
│   └── __init__.py
├── requirements.txt            # pip 依赖
└── README.md                   # 本文
```

> **权重不放进本包**（约 6.7 GB）。请单独把 `nuwa_weights/` 传到服务器，
> 并放在 `nuwa_agent/` 的**上级目录**（即 `nuwa_server_upload/nuwa_weights/`），
> 或任意目录并用环境变量 `NUWA_BASE_DIR` 指定。

---

## 2. 上传到服务器

任选一种：

```bash
# 方式 A：scp 整个文件夹
scp -r nuwa_server_upload/ user@server:/your/workdir/

# 方式 B：rsync（断点续传，适合大目录）
rsync -avz nuwa_server_upload/ user@server:/your/workdir/nuwa_server_upload/
```

权重（单独传，示例放在上级目录）：

```bash
rsync -avz /d/thesis/NUWA-main\ \(1\)/nuwa_weights/ \
      user@server:/your/workdir/nuwa_server_upload/nuwa_weights/
```

---

## 3. 环境准备（服务器）

```bash
# 3.1 新建 conda 环境（推荐，便于装 ViennaRNA）
conda create -n nuwa python=3.10 -y
conda activate nuwa

# 3.2 pip 依赖
cd /your/workdir/nuwa_server_upload
pip install -r requirements.txt

# 3.3 外部生物信息学工具
conda install -c bioconda viennarna -y     # RNA.fold 折叠
pip install cai2                            # CAI 计算

# 3.4 设置智谱 AI key（务必，config.py 不再写死明文）
export NUWA_API_KEY="你的智谱AI key"
```

> 未安装外部工具时 `evaluator.py` 会自动回退到启发式估算，代码可跑但分数不精确。
> LLM 调用需要服务器能访问 `https://open.bigmodel.cn`（出网权限）。

---

## 4. 路径配置（两种写法，二选一）

权重位置由 `nuwa_agent/config.py` 里的 `BASE_DIR` 决定：

- **默认**：`BASE_DIR` = `nuwa_agent/` 的父目录。
  也就是说只要把 `nuwa_weights/` 放在 `nuwa_server_upload/nuwa_weights/` 下，无需任何改动即可运行。

- **自定义**：若权重放在别处，设环境变量即可，不用改文件：

  ```bash
  export NUWA_BASE_DIR=/abs/path/to/parent_of_nuwa_weights
  ```

config.py 期望的权重布局：

```
$NUWA_BASE_DIR/nuwa_weights/
├── domain_models/{bacteria,eukaryote,archaea}/checkpoint-*
├── finetuned/{te,stability,expression}/checkpoint-1000
├── species_maps/{bacteria,eukaryote,archaea}_species_mapping.json
└── codon_tables/*.json
```

---

## 5. 运行

```bash
cd /your/workdir/nuwa_server_upload/nuwa_agent
export NUWA_API_KEY="你的智谱AI key"     # 若还没设

# 5.1 一键 demo（EGFP 30aa + E.coli，3 轮快速验证）
python run_demo.py

# 5.2 正式运行：种子与目标特异 MFE/nt 窗口均为必填
export NUWA_RUN_SEED=20260918
export NUWA_MFE_PER_NT_MIN=-0.35
export NUWA_MFE_PER_NT_MAX=-0.20
# 正式模式默认对生成模型和三个回归 checkpoint 的所有文件做 SHA256；请勿关闭。
export NUWA_AUDIT_HASH_MODELS=1
python main.py
#   依次输入：蛋白序列(FASTA或裸序列) → 宿主名(如 Escherichia coli) → GC 约束(可留空)

# 非交互式单条运行（推荐同时提供稳定 protein-id）
python main.py --protein-file /path/one_protein.fasta --protein-id NP_000000.1 \
  --host "Escherichia coli" --gc-min 0.3 --gc-max 0.7

# 可选：运行确定性基准策略（默认启用多 Agent 讨论）
export NUWA_MULTIAGENT_ENABLED=0
python main.py
```

Phase 2 每轮先执行上一轮选定的温度和同义密码子替换策略，再生成、评分、检查约束并计算 Pareto 前沿与 HV。三个专家 Agent 分别讨论生物约束、生成探索和同义编辑，经过质疑与修订后由 Central LLM 给出下一轮的结构化决策：生成温度、新生成/突变比例、每条突变子代的同义替换次数及精英父本选择。代码会校验参数、限制策略单轮跳变，并在持续退化时启用保守恢复；CAI 调整仍按确定性计划执行。单个讨论阶段会独立重试，未通过校验的原文不会传给后续 Agent；Central LLM 失败时优先从已验证的专家结论合成安全决定，没有足够有效结论时才回退到预设策略。当前轮 HV 明显低于历史最优时继续使用历史最优前沿作父本。硬约束、评分、Pareto 选择及 HV 计算均由程序完成。`NUWA_MULTIAGENT_ENABLED=0` 可关闭讨论，运行相同的基准策略。

输出包括：`./output/nuwa_agent_<时间戳>.json`（机器可读完整记录）、`nuwa_agent_reviewer_<时间戳>.md`（审稿人优先阅读的精简报告）和 `nuwa_agent_chain_<时间戳>.md`（完整 prompt/response 审计附录）。结果采用原子写入，不会留下半截 JSON/Markdown。正式模式必须设置 `NUWA_RUN_SEED`；否则程序会在生成前终止。

JSON 的 `run_metadata` 保存完整输入蛋白、蛋白 SHA256、宿主、随机种子、Python/依赖版本、全部 Python 源码指纹、密码子表指纹以及四套模型制品的逐文件 SHA256 和 manifest SHA256。默认在严格模式下启用模型内容哈希；大型 checkpoint 第一次计算可能需要一些时间。最终 `pareto_solutions[].sequence` 是无空格大写 RNA，`sequence_codon_spaced` 仅供阅读，`sequence_length` 始终按无空格序列计算。原始生成历史为了审计可能继续保留密码子分隔格式。

保存前程序还会生成 `artifact_validation`，自动检查输入哈希、模型制品哈希、物种 class ID、逐轮决策衔接、历史最佳轮来源、规范序列、终止密码子和全部 active hard constraints。正式结果应满足 `artifact_validation.passed=true`；若为 false，控制台会打印 `AUDIT WARNING`，该次结果不得作为正式实验有效结果。

`iteration_history` 明确区分：`central_requested_decision`（LLM 对下一轮的原始请求）、`guardrail_adjusted_decision`（控制器校验后的下一轮参数）和 `applied_decision`（生成当前轮时真实执行的参数）；`decision_for_round` 给出决定适用的轮次。自由文本理由只保留在原始请求中，权威执行摘要由最终数值自动生成，避免护栏调整后文字与字段矛盾。物种元数据同时记录最终代理和 `resolver_match` 初始建议；MFE 报告明确区分 active 与 inactive legacy bounds；精英突变记录包含同义突变、CAI 后处理和最终评估序列的完整 provenance chain。`optimization_result.best_hv` 是返回解对应的历史最优 HV，`last_round_hv` 是真实末轮 HV。输出不包含 `NUWA_API_KEY`。固定种子控制本地 Python/NumPy/PyTorch 随机过程；远端 LLM 响应及某些 GPU 运算仍可能存在差异，因此完整请求与响应也会保存在审计附录中。

---

## 6. 常见问题

| 现象 | 原因 / 处理 |
|------|------------|
| `请先设置环境变量 NUWA_API_KEY` | 没 export key，先 `export NUWA_API_KEY=...` |
| 模型权重 `FileNotFoundError` | `BASE_DIR` 指错或 `nuwa_weights/` 没传；检查第 4 节布局 |
| `No module named 'RNA'` / `cai2` | 外部工具未装；按 3.3 安装；正式实验模式会立即停止 |
| 连不上 `open.bigmodel.cn` | 服务器无出网权限，联系管理员放通 443 |
| 某轮讨论失败 | 查看输出中的 `discussion_fallback_used` 和 `discussion_error`；下一轮会采用预设的确定性策略 |
| CUDA OOM | 减小 `config.NUM_CANDIDATES` 或用 CPU（改 `device`） |

---

## 7. 与本地旧版的差异（迁移说明）

- 旧 `config.py` 写死 `D:/thesis/...` Windows 路径 → 本包改为 `BASE_DIR` 派生 + 环境变量。
- 旧 `config.example.py` 指向已删除的 `virtual-lab-main/NUWA-*-model`、`finetuned_model_*` → 本包已废弃该模板，路径统一走 `nuwa_weights/`。
- Phase 2 已加入专家讨论与 Central LLM 决策。新旧策略、运行种子与每轮实际参数需要分别记录，重跑结果不应与旧结果视为同一实验配置。
