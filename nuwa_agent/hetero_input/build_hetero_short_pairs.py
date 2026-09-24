"""Generate two length-matched FASTA files for hetero-expression experiments.

File A: 100 short HUMAN proteins (from GCF_000001405.40_protein.faa), amino-acid
        sequences, lengths chosen to sit in the short end of the E.coli length
        distribution (so NUWA can redesign them toward E.coli).
File B: 100 E.coli CDS (from Ecoli_CDS.fasta), DNA, each greedy length-matched to
        the SAME target length as its human counterpart -> the two files are
        length-aligned (max |Δ| codon small), both short.

Method: pick 100 target lengths evenly spread across the short window of the
E.coli codon-length distribution [p10, p50]; for each target, independently pick
the closest unused human protein and the closest unused E.coli CDS.
"""
import os
import re
import numpy as np
from scipy.stats import ks_2samp

HUMAN_FAA = r"C:/Users/30778/xwechat_files/wxid_nvws1boumcws12_e58a/msg/file/2026-07/GCF_000001405.40_protein.faa"
ECO_FASTA = r"C:/Users/30778/xwechat_files/wxid_nvws1boumcws12_e58a/msg/file/2026-07/Ecoli_CDS.fasta"
OUT_HUMAN = r"D:/thesis/NUWA-main (1)/hetero_analysis/human_short_ecolilen_100.fasta"
OUT_ECO = r"D:/thesis/NUWA-main (1)/hetero_analysis/ecoli_matched_100.fasta"

STD = set("ACDEFGHIKLMNPQRSTVWY")
N_TARGET = 100


def parse_fasta(path):
    sid = None
    seq = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if sid:
                    yield sid, "".join(seq)
                sid = line[1:].rstrip("\n")
                seq = []
            else:
                seq.append(line.strip())
    if sid:
        yield sid, "".join(seq)


def trim_stop(s):
    s = s.rstrip("N")
    if len(s) >= 3 and s.endswith(("TAA", "TAG", "TGA")):
        s = s[:-3]
    return s


# ---- E.coli CDS (DNA) ----
eco = []
for h, s in parse_fasta(ECO_FASTA):
    s = s.upper().replace("U", "T")
    if "N" in s:
        continue
    s = trim_stop(s)
    cl = len(s) // 3
    if 30 <= cl <= 600:
        eco.append((h.split()[0], s, cl))
eco_arr = np.array([e[2] for e in eco])
print(f"[ecoli] usable CDS (30-600 codon): n={len(eco)}  median={int(np.median(eco_arr))} "
      f"p10={int(np.percentile(eco_arr,10))} p50={int(np.percentile(eco_arr,50))}")

# ---- Human proteins (amino acid), prefer curated NP_ ----
hum = []
for h, s in parse_fasta(HUMAN_FAA):
    s = s.upper()
    if any(c not in STD for c in s):      # drop X/U/B/Z/* etc.
        continue
    pid = re.match(r"^(NP_\d+\.\d+|XP_\d+\.\d+)", h)
    if not pid:
        continue
    aa = len(s)
    if aa < 60 or aa > 450:               # short window
        continue
    is_np = h.startswith("NP_")
    name = h.split(" ", 1)[1] if " " in h else ""
    hum.append((pid.group(1), name, s, aa, is_np))
hum_arr = np.array([x[3] for x in hum])
print(f"[human] candidates (60-450 aa, std-20): n={len(hum)}  "
      f"NP_={int(np.sum([x[4] for x in hum]))}  XP_={len(hum)-int(np.sum([x[4] for x in hum]))}")

# ---- 100 short target lengths, spread across E.coli short window ----
lo, hi = np.percentile(eco_arr, 10), np.percentile(eco_arr, 50)
targets = [int(round(t)) for t in np.linspace(lo, hi, N_TARGET)]
targets = [max(60, min(450, t)) for t in targets]
print(f"[targets] n={len(targets)} range=[{min(targets)}, {max(targets)}]")

eco_used = [False] * len(eco)
hum_used = [False] * len(hum)
sel_hum, sel_eco = [], []

for t in targets:
    # closest unused E.coli CDS
    dei = np.argsort(np.abs(eco_arr - t))
    ei = next((i for i in dei if not eco_used[i]), None)
    if ei is None:
        continue
    eco_used[ei] = True
    # closest unused human protein
    dhi = np.argsort(np.abs(hum_arr - t))
    hi_i = next((i for i in dhi if not hum_used[i]), None)
    if hi_i is None:
        eco_used[ei] = False
        continue
    hum_used[hi_i] = True
    sel_eco.append(eco[ei])
    sel_hum.append(hum[hi_i])

print(f"[selected] human={len(sel_hum)} ecoli={len(sel_eco)}")

# ---- write ----
os.makedirs(os.path.dirname(OUT_HUMAN), exist_ok=True)
with open(OUT_HUMAN, "w") as f:
    for pid, name, seq, aa, _ in sel_hum:
        f.write(f">{pid} {name}\n")
        for i in range(0, len(seq), 70):
            f.write(seq[i:i + 70] + "\n")
with open(OUT_ECO, "w") as f:
    for sid, seq, cl in sel_eco:
        f.write(f">{sid} codon_len={cl}\n")
        for i in range(0, len(seq), 70):
            f.write(seq[i:i + 70] + "\n")

# ---- verify match quality ----
hl = np.array([x[3] for x in sel_hum])
el = np.array([x[2] for x in sel_eco])
print(f"[verify] human  len min={hl.min()} median={int(np.median(hl))} max={hl.max()}")
print(f"[verify] ecoli  len min={el.min()} median={int(np.median(el))} max={el.max()}")
print(f"[verify] max|human-ecoli| codon = {int(np.abs(hl-el).max())}")
print(f"[verify] KS D(human,ecoli) = {ks_2samp(hl, el).statistic:.3f}")
print(f"[verify] written:\n  {OUT_HUMAN}\n  {OUT_ECO}")
