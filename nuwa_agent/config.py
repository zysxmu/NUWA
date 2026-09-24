"""NUWA-Agent v6 配置 — 服务器部署版。

====================================================================
路径策略（关键：本文件不再写死 Windows 盘符路径）
--------------------------------------------------------------------
所有权重 / 物种映射 / 密码子表都放在一个 ``nuwa_weights/`` 目录下，
其结构固定为::

    nuwa_weights/
    ├── domain_models/
    │   ├── bacteria/checkpoint-930760
    │   ├── eukaryote/checkpoint-1153028
    │   └── archaea/checkpoint-59820
    ├── finetuned/
    │   ├── te/checkpoint-1000
    │   ├── stability/checkpoint-1000
    │   └── expression/checkpoint-1000
    ├── species_maps/{bacteria,eukaryote,archaea}_species_mapping.json
    └── codon_tables/*.json

``BASE_DIR`` 指向「包含 nuwa_weights/ 的上级目录」：
  * 默认 = 本文件所在 nuwa_agent/ 的父目录
    （即把 nuwa_server_upload/ 作为根，nuwa_weights/ 放在
     nuwa_server_upload/nuwa_weights/ 即可直接生效）；
  * 也可用环境变量覆盖：``export NUWA_BASE_DIR=/your/path``。

API key 策略（安全：绝不写死明文）
--------------------------------------------------------------------
从环境变量 ``NUWA_API_KEY`` 读取。部署前先::

    export NUWA_API_KEY="你的智谱AI key"

未设置会在 import 时直接报错并给出提示，避免无声失败。
====================================================================
"""

import os

# ===================== 只需改这里 =====================
# 指向包含 nuwa_weights/ 的目录（或用环境变量 NUWA_BASE_DIR 覆盖）
BASE_DIR = os.environ.get(
    "NUWA_BASE_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
# =====================================================

# ---- LLM ----
LLM_API_KEY = os.environ.get("NUWA_API_KEY", "")
if not LLM_API_KEY:
    raise RuntimeError(
        "请先设置环境变量 NUWA_API_KEY：\n"
        "    export NUWA_API_KEY='你的智谱AI key'\n"
        "（config.py 不留存明文 key，避免泄露）"
    )
LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
LLM_MODEL = "GLM-5.1"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 16384

# 专家讨论默认启用；关闭时采用预注册的固定搜索策略，便于离线测试。
MULTIAGENT_ENABLED = os.environ.get("NUWA_MULTIAGENT_ENABLED", "1").strip().lower() not in {
    "0", "false", "no", "off"
}
# 正式实验默认禁止任何启发式回退。Demo 会显式关闭。
STRICT_EVALUATION = os.environ.get("NUWA_STRICT_EVALUATION", "1").strip().lower() not in {
    "0", "false", "no", "off"
}

# ---- NUWA 模型 ----
NUWA_MODELS = {
    "bacteria": {
        "name": "NUWA-Bacteria",
        "path": os.path.join(BASE_DIR, "nuwa_weights", "domain_models", "bacteria", "checkpoint-930760"),
        "domain": "Bacteria",
        "num_species": 19676,  # class_id 范围: 0-19675
        "description": "细菌 mRNA 设计模型，擅长原核生物密码子优化",
        "strengths": [
            "原核生物密码子适应指数 (CAI) 优化",
            "细菌 mRNA 稳定性预测 (TE 预测 SOTA)",
            "大肠杆菌等模式生物的高表达设计",
            "GC 含量中等 (50-55%) 的序列优化"
        ],
        "limitations": [
            "对真核生物 poly-A 信号和 UTR 设计较弱",
            "不擅长处理内含子相关的优化",
            "极端 GC 环境 (< 30% 或 > 70%) 表现下降"
        ]
    },
    "eukaryote": {
        "name": "NUWA-Eukaryote",
        "path": os.path.join(BASE_DIR, "nuwa_weights", "domain_models", "eukaryote", "checkpoint-1153028"),
        "domain": "Eukaryote",
        "num_species": 4688,  # class_id 范围: 0-4687
        "description": "真核生物 mRNA 设计模型，擅长哺乳动物密码子优化",
        "strengths": [
            "哺乳动物密码子适应指数 (CAI) 优化",
            "真核 mRNA 稳定性和半衰期预测 (SOTA)",
            "人类/小鼠等模式生物的高表达设计",
            "5' UTR / 3' UTR 序列优化",
            "GC 含量偏高 (55-65%) 的序列优化"
        ],
        "limitations": [
            "对原核生物 RBS 序列设计无专门优化",
            "低 GC 环境 (< 40%) 表现一般",
            "古菌特异性密码子使用模式覆盖不足"
        ]
    },
    "archaea": {
        "name": "NUWA-Archaea",
        "path": os.path.join(BASE_DIR, "nuwa_weights", "domain_models", "archaea", "checkpoint-59820"),
        "domain": "Archaea",
        "num_species": 702,  # class_id 范围: 0-701
        "description": "古菌 mRNA 设计模型，擅长极端环境密码子优化",
        "strengths": [
            "极端环境 (高温/高盐) 密码子优化",
            "古菌 mRNA 结构稳定性预测",
            "极端 GC 含量 (> 70% 或 < 30%) 的序列优化",
            "嗜热/嗜盐菌的 mRNA 设计"
        ],
        "limitations": [
            "对常见模式生物 (大肠杆菌/人类) 不如专域模型",
            "中温中盐环境无明显优势",
            "训练数据量相对较少，泛化性可能受限"
        ]
    }
}

# ---- 生成参数 ----
NUM_CANDIDATES = 50           # 每轮生成候选数
GENERATION_TOP_K = 50
GENERATION_TOP_P = 0.95

# ---- 默认约束 ----
DEFAULT_CONSTRAINTS = {
    "cai_min": 0.7,            # CAI 下限
    "gc_min": 0.30,            # GC% 下限
    "gc_max": 0.70,            # GC% 上限
    "mfe_max": -100,           # MFE 上限 (kcal/mol, 负值, 要求结构有一定稳定性)
    "mfe_min": -400,           # MFE 下限 (kcal/mol, 防止过度折叠 — Stability 模型惩罚过负 MFE)
    "max_stem_length": 33,     # 最大茎区长度 (bp)
    "max_homopolymer": 6,      # 允许 ≤5 连同聚物（714nt 序列约 84% 满足；放宽需同步 Methods 表）
}

# ---- Pareto ----
PARETO_TOP_K = 10              # Pareto 前沿保留数

# ---- 迭代优化 ----
MAX_ITERATION_ROUNDS = 10          # 最大迭代轮数
MIN_ITERATION_ROUNDS = 5            # 最少迭代轮数 (防止过早收敛)
HV_CONVERGENCE_THRESHOLD = 0.01    # 稳定轮要求 |ΔHV| < 0.01
HV_CONVERGENCE_PATIENCE = 2        # 连续稳定轮数；避免一次偶然小回升触发收敛
HV_BEST_GAP_TOLERANCE = 0.01       # 当前 HV 必须接近历史最优，才允许判定收敛

# ---- CoT 推理链 ----
COT_VERBOSE = True             # 打印每步推理摘要

# ---- 外部工具路径 ----
CODON_TABLES_DIR = os.path.join(BASE_DIR, "nuwa_weights", "codon_tables")       # 宿主参考密码子表目录

# ---- 微调回归预测模型路径 ----
# 三个 NUWA 微调回归模型，用于替换 evaluator.py 中的启发式软目标
# 架构: BertForRegressionHF = BertModel(NUWA backbone) + Linear(768, 1)
FINETUNED_TE_MODEL = os.path.join(BASE_DIR, "nuwa_weights", "finetuned", "te", "checkpoint-1000")
# TE 预测模型（Spearman≈0.484，bacteria backbone，type_vocab=19676）

FINETUNED_STABILITY_MODEL = os.path.join(BASE_DIR, "nuwa_weights", "finetuned", "stability", "checkpoint-1000")
# Stability 预测模型 — all domains（Spearman≈0.758，type_vocab=19676）

FINETUNED_EXPR_MODEL = os.path.join(BASE_DIR, "nuwa_weights", "finetuned", "expression", "checkpoint-1000")
# Expression 预测模型 — all domains（Spearman≈0.768，type_vocab=4688）

# ---- 物种映射表 ----
SPECIES_MAPS_DIR = os.path.join(BASE_DIR, "nuwa_weights", "species_maps")
# 映射文件格式: {domain}_species_mapping.json
#   bacteria_species_mapping.json  (19676 species)
#   eukaryote_species_mapping.json (4688 species)
#   archaea_species_mapping.json   (702 species)
# 文件内部格式: {"species_to_id": {"Escherichia coli": 5834, ...}, "id_to_species": {...}}

# ---- 输出 ----
OUTPUT_DIR = "./output"
