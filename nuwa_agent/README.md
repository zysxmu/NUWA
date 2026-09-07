# NUWA-Agent

NUWA-Agent 是一个面向 **mRNA / CDS 多目标序列设计** 的智能体系统。它把预训练的
NUWA masked-language 模型（按物种域分 bacteria / eukaryote / archaea 三个检查点）
与一个大语言模型（LLM）驱动的「链式思考（CoT）决策」以及「迭代式多目标优化」闭环结合起来，
为给定目标蛋白自动设计出在 **翻译效率（TE）/ 结构稳定性（Stability）/ 表达量（Expression）**
三个目标上取得帕累托最优、且满足密码子偏好、GC、二级结构、同聚物与免疫原性等多项硬约束的编码序列。

## 工作流概览

1. **Phase 0 — SpeciesResolver**：根据宿主物种名解析出 `(domain, class_id, confidence)`
   （四级查找：精确 / 属级 / 模糊 / 默认）。
2. **Phase 1 — Orchestrator（LLM CoT 三层决策）**：域级 → 物种级 → 跨域级，
   选定 NUWA 域模型 + `class_id` + 约束边界，并产出可读的推理链。
3. **Phase 2 — IterationController（迭代优化，5–10 轮）**：
   生成 → 外部生物信息学工具评估 → 约束判定 → NSGA-II 帕累托筛选 →
   LLM 反馈（结构化指令）→ 收敛判定。温度前 40% 轮次从 1.0 衰减到 0.3。
4. **Phase 3 — 输出**：帕累托最优序列、好序列队列、完整推理链 Markdown 与全部迭代历史 JSON。

> **关于 LLM 反馈的说明**：NUWA 是 masked-LM，没有自然语言 prompt 接口，
> 因此 LLM 的自然语言分析会被「编译」为 4 个结构化旋钮
> （`explore` 温度 / `cai_intensity` 同义替换强度 / `mutate_fraction` 精英替换比例 /
> `focus_objective` 弱目标偏置）来驱动下一轮生成。自然语言反馈本身不进入生成模型。

## 目录结构

```
nuwa_agent/
├── config.example.py          # 配置模板（复制为 config.py 后填写你自己的 key/路径）
├── main.py                    # 入口：交互式运行完整流程
├── run_demo.py                # 快速演示（30aa EGFP 片段，3 轮迭代）
├── orchestrator.py            # Phase 1：SpeciesResolver + LLM CoT 三层决策
├── species_resolver.py        # 物种名 → (domain, class_id, confidence)
├── model_registry.py          # 加载 NUWA 域模型 + 熵引导生成逻辑
├── evaluator.py               # 3 目标（TE/Stability/Expression）+ 6 维硬约束评估
├── constraint_checker.py      # 约束可行性判定（MFE 双侧 + 免疫原性一票否决）
├── pareto_selector.py         # NSGA-II 非支配排序 + 拥挤度 + Hypervolume
├── feedback_analyzer.py       # LLM 分析帕累托前沿 → 生成结构化改进指令
├── iteration_controller.py    # Phase 2：迭代优化主循环（温度衰减/收敛/突变）
└── .gitignore                 # 已忽略 config.py / output / data / logs / 缓存
```

## 环境要求

### Python 依赖

```bash
pip install openai>=1.0.0 transformers torch tokenizers tqdm numpy scipy safetensors pymoo cai2 mhcflurry
```

- `ViennaRNA`（二级结构 MFE 计算，evaluator 用）建议用 conda 安装：
  ```bash
  conda install -c bioconda viennarna
  ```
- 三个目标（TE / Stability / Expression）由「微调 NUWA 回归模型」打分，
  通过 `transformers` 加载（需要 `torch` + 可用的 GPU，推荐 CUDA）。

### 需要的模型权重与外部数据（不在本仓库内）

本仓库**只包含 Agent 代码**，以下几类大体积 / 训练产物需另行准备：

| 资源 | 配置项 | 说明 |
|------|--------|------|
| NUWA 三域模型（bacteria / eukaryote / archaea） | `NUWA_MODELS[...]["path"]` | BertForMaskedLM safetensors 检查点 |
| 微调回归模型 ×3（TE / Stability / Expression） | `FINETUNED_TE_MODEL` / `FINETUNED_STABILITY_MODEL` / `FINETUNED_EXPR_MODEL` | BertForRegressionHF 检查点 |
| 物种映射表（species_to_id / id_to_species） | `SPECIES_MAPS_DIR` | 位于 eukaryote 模型目录下 |
| 宿主参考密码子表（JSON，如 `escherichia_coli.json` / `homo_sapiens.json`） | `CODON_TABLES_DIR` | 用于 CAI 计算（cai2） |

> 这些权重由各自的训练流程产出，体积较大且含训练数据，请按你实验室的路径存放，
> 并在 `config.py` 中指向对应位置（见下）。

## 配置（必须）

1. 复制配置模板（**不要直接改 `config.example.py`**，它被提交进仓库；
   真正的 `config.py` 已被 `.gitignore` 忽略，不会上传）：

   ```bash
   cp config.example.py config.py
   ```

2. 编辑 `config.py`，至少修改以下项：

   - `LLM_API_KEY`：改成你自己的智谱 AI（Zhipu）API key（形如 `xxxx.nvjCrxOMnsXMmsrx`）。
   - `NUWA_MODELS` 中三个域的 `"path"`：指向你本地存放的 NUWA 域模型检查点目录。
   - `FINETUNED_TE_MODEL` / `FINETUNED_STABILITY_MODEL` / `FINETUNED_EXPR_MODEL`：
     指向三个微调回归模型检查点。
   - `SPECIES_MAPS_DIR`：物种映射表所在目录。
   - `CODON_TABLES_DIR`：宿主参考密码子表所在目录（CAI 计算用）。

3. 其余参数（候选数、约束默认值、迭代轮数、MHC 等位基因、温度衰减等）
   一般无需改动，详见 `config.example.py` 内注释。

## 运行

### 交互式（完整流程）

```bash
python main.py
```

按提示依次输入：
- **目标蛋白序列**（氨基酸序列，如 `MVSKGEELFTGVVPILVELDGDVNGHKFSVS`）
- **宿主物种**（如 `Escherichia coli`、`Homo sapiens`）
- **GC 含量约束**（可选，如 `0.3-0.7`；直接回车跳过用默认约束）

### 快速演示

```bash
python run_demo.py
```

使用内置的 30aa EGFP 片段 + *E. coli* 宿主，并将候选数/轮数调小以便快速验证环境。

## 输出

运行结果写入 `./output/`（**自动生成，已被 `.gitignore` 忽略，不会上传**）：

- `nuwa_agent_<时间戳>.json` — 完整结果（模型选择、帕累托解、每轮迭代历史、LLM 日志，**不截断**）
- `nuwa_agent_chain_<时间戳>.md` — 可读的完整推理链 & 优化过程报告

## 本仓库**不**包含的内容（已被 `.gitignore` 忽略）

- `config.py`（含你的 API key，**绝不提交**）
- `./output/`、`data/`、`results/`（运行结果、中间数据）
- `*.log`、`*.csv`、`*.jsonl`（日志与导出数据）
- `__pycache__/`、`*.pyc`、`*.bak*`（Python 缓存与备份）
- 任意模型权重 / 训练数据（需另备，见上文）

## 常见问题

- **`ModuleNotFoundError: No module named 'viennarna'` / `cai2` / `mhcflurry`**：
  这些外部生物信息学工具未安装，按「环境要求」一节补齐。
- **`Permission denied` 或 `remote: Repository not found`**：
  这是把代码推到 GitHub 时的问题，与运行无关；请确认你已按 `config.py` 配置好本地路径。
- **生成很慢 / 显存不足**：NUWA 域模型与微调回归模型在 GPU 上推理，
  确认 CUDA 可用，或减小 `NUM_CANDIDATES` / `MAX_ITERATION_ROUNDS`（参考 `run_demo.py`）。
