"""NUWA-Agent v6 配置模板 — 复制本文件为 config.py 并填入你自己的 key。

用法：
    cp config.example.py config.py
    # 然后编辑 config.py，把 LLM_API_KEY 改成你自己的智谱 AI key
（config.py 已在 .gitignore 中，不会被提交到 GitHub）
"""

# ---- LLM ----
# TODO: 替换成你自己的智谱 AI API key（形如 xxxxxxxx.nvjCrxOMnsXMmsrx）
LLM_API_KEY = "YOUR_ZHIPU_API_KEY_HERE"
LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
LLM_MODEL = "GLM-5.1"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 16384

# ---- NUWA 模型 ----
NUWA_MODELS = {
    "bacteria": {
        "name": "NUWA-Bacteria",
        "path": "D:/thesis/virtual-lab-main/NUWA-bacteria-model/NUWA-bacteria-model/checkpoint-930760",
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
        "path": "D:/thesis/virtual-lab-main/NUWA-eukaryote-model/NUWA-eukaryote-model/checkpoint-1153028",
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
        "path": "D:/thesis/virtual-lab-main/NUWA-archaea-model/NUWA-archaea-model/checkpoint-59820",
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
    "max_homopolymer": 10,     # 最大同聚物长度 (nt) — 2026-08-22 由 4 放宽至 10
    "safety_threshold": 0.55,   # 免疫原性风险阈值 (0.55≈500nM weak binder cutoff, P95聚合)
}

# ---- Pareto ----
PARETO_TOP_K = 10              # Pareto 前沿保留数

# ---- 迭代优化 ----
MAX_ITERATION_ROUNDS = 10          # 最大迭代轮数
MIN_ITERATION_ROUNDS = 5            # 最少迭代轮数 (防止过早收敛)
HV_CONVERGENCE_THRESHOLD = 0.01    # HV 改进 < 1% → 收敛 (仅在 MIN_ITERATION_ROUNDS 之后生效)

# ---- CoT 推理链 ----
COT_VERBOSE = True             # 打印每步推理摘要

# ---- 免疫原性等位基因 ----
DEFAULT_MHC_ALLELES = [
    "HLA-A*02:01", "HLA-A*24:02", "HLA-B*07:02",
    "HLA-B*40:01", "HLA-C*07:02",  # class1 旧模型不支持 C*07:01, 改用 C*07:02
]

# ---- 外部工具路径 ----
CODON_TABLES_DIR = "./codon_tables"       # 宿主参考密码子表目录

# ---- 微调回归预测模型路径 ----
FINETUNED_TE_MODEL = "D:/thesis/NUWA-main (1)/finetuned_model_TE/checkpoint-1000"
FINETUNED_STABILITY_MODEL = "D:/thesis/NUWA-main (1)/finetuned_model_fungal/checkpoint-1000"
FINETUNED_EXPR_MODEL = "D:/thesis/NUWA-main (1)/finetuned_model_fungal_euk/checkpoint-1000"

# ---- 物种映射表 ----
SPECIES_MAPS_DIR = "D:/thesis/virtual-lab-main/NUWA-eukaryote-model/NUWA-eukaryote-model"

# ---- 输出 ----
OUTPUT_DIR = "./output"
