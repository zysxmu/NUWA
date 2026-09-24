"""Batch driver for NUWA-Agent: optimize many proteins from one FASTA file.

Why this exists
--------------
The server code's `--protein-file` reads the WHOLE file as a SINGLE concatenated
sequence (see main.py `_read_protein_fasta`: it drops `>` headers and joins all
sequence lines). So you CANNOT point `--protein-file` at a multi-record FASTA.

This driver does NOT modify any run code. It:
  1. splits a multi-record protein FASTA into one temp file per record;
  2. calls the EXISTING `main.py` once per record (same verified single-run path);
  3. merges one preferred feasible rank-0 Pareto solution per run into `batch_best_sequences.fasta`,
     with headers the hetero_analysis scripts already parse
     (`>NP_xxx_rank1 host=... model=... TE=... ...`).

It is resumable: proteins whose pid already appears in the merged output are skipped.

Usage (on the server, inside nuwa_agent/):
    export NUWA_API_KEY=... NUWA_BASE_DIR=/root/autodl-tmp
    export NUWA_MFE_PER_NT_MIN=-0.35 NUWA_MFE_PER_NT_MAX=-0.20
    python batch_run.py --protein-file /path/human_short_ecolilen_100.fasta \
                        --host "Escherichia coli" --gc-min 0.3 --gc-max 0.7

Test pieces without the LLM:
    python batch_run.py --protein-file X.fasta --only-split
    python batch_run.py --only-collect
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

from config import OUTPUT_DIR

STD_AA = set("ACDEFGHIKLMNPQRSTVWY")
HERE = os.path.dirname(os.path.abspath(__file__))
MERGED_NAME = "batch_best_sequences.fasta"


def parse_multi_fasta(path):
    """Return list of (header_id, full_header, sequence) for every record."""
    records = []
    cur_id = None
    cur_header = None
    seq = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if cur_id is not None:
                    records.append((cur_id, cur_header, "".join(seq)))
                cur_header = line[1:].strip()
                cur_id = cur_header.split()[0] if cur_header else f"rec{len(records)}"
                seq = []
            else:
                seq.append(line.strip())
    if cur_id is not None:
        records.append((cur_id, cur_header, "".join(seq)))
    return records


def normalize(seq):
    seq = re.sub(r"\s+", "", seq).upper()
    if seq.endswith("*"):
        seq = seq[:-1]
    invalid = sorted(set(seq) - STD_AA)
    if invalid:
        raise ValueError(f"unsupported residues: {''.join(invalid)}")
    return seq


def split_fasta(path, tmp_dir):
    os.makedirs(tmp_dir, exist_ok=True)
    written = []
    for i, (pid, header, seq) in enumerate(parse_multi_fasta(path)):
        seq = normalize(seq)
        out = os.path.join(tmp_dir, f"rec_{i:03d}.fasta")
        with open(out, "w") as f:
            f.write(f">{header}\n")
            for j in range(0, len(seq), 70):
                f.write(seq[j:j + 70] + "\n")
        written.append((pid, out))
    return written


def run_one(protein_file, host, gc_min, gc_max, pid):
    """Call the existing main.py for ONE record; return the new JSON path."""
    before = set(glob.glob(os.path.join(OUTPUT_DIR, "nuwa_agent_*.json")))
    cmd = [sys.executable, os.path.join(HERE, "main.py"),
           "--protein-file", protein_file, "--protein-id", pid, "--host", host]
    if gc_min is not None:
        cmd += ["--gc-min", str(gc_min), "--gc-max", str(gc_max)]
    env = dict(os.environ)
    env["PYTHONPATH"] = HERE + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + "\n" + r.stderr + "\n")
        raise RuntimeError(f"main.py failed for {protein_file}")
    after = set(glob.glob(os.path.join(OUTPUT_DIR, "nuwa_agent_*.json"))) - before
    new_json = after.pop() if after else None
    return new_json


def best_rank0(json_path):
    """Return one preferred feasible member of the rank-0 Pareto front."""
    with open(json_path) as f:
        data = json.load(f)
    sols = data.get("pareto_solutions", [])
    r1 = [s for s in sols if s.get("rank") == 0]
    if not r1:
        return None, None
    feas = [s for s in r1 if s.get("is_feasible")]
    pool = feas if feas else r1
    best = max(pool, key=lambda s: s["objectives"]["Expression"])
    model = data.get("model_selection", {}).get("selected_model", "bacteria")
    return model, best


def collect(merged_path, pids_seen):
    """Merge one preferred rank-0 Pareto solution from each run into FASTA."""
    jsons = sorted(glob.glob(os.path.join(OUTPUT_DIR, "nuwa_agent_*.json")))
    written = 0
    with open(merged_path, "w") as out:
        for jp in jsons:
            with open(jp) as f:
                data = json.load(f)
            pid_raw = (data.get("run_metadata", {}).get("input", {}).get("protein_id")
                       or data.get("input_protein_id", ""))
            m = re.search(r"(NP_\d+\.\d+|XP_\d+\.\d+)", pid_raw)
            if not m:
                continue
            pid = m.group(1)
            if pid in pids_seen:
                continue
            model, sol = best_rank0(jp)
            if sol is None:
                continue
            seq = sol["sequence"].replace(" ", "").upper().replace("T", "U")
            pids_seen.add(pid)
            o = sol["objectives"]
            c = sol["constraints"]
            out.write(
                f">{pid}_rank1 host=Escherichia coli model={model} "
                f"TE={o['TE']:.4f} Stab={o['Stability']:.4f} Expr={o['Expression']:.4f} "
                f"CAI={c['CAI']:.4f} GC={c['GC%']:.3f} MFE={c['MFE']:.1f} "
                f"feasible={str(sol['is_feasible']).lower()}\n"
            )
            for i in range(0, len(seq), 70):
                out.write(seq[i:i + 70] + "\n")
            written += 1
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein-file", required=False)
    ap.add_argument("--host", default="Escherichia coli")
    ap.add_argument("--gc-min", type=float, default=None)
    ap.add_argument("--gc-max", type=float, default=None)
    ap.add_argument("--tmp-dir", default=os.path.join(HERE, "batch_tmp"))
    ap.add_argument("--merged", default=os.path.join(HERE, MERGED_NAME))
    ap.add_argument("--only-split", action="store_true")
    ap.add_argument("--only-collect", action="store_true")
    args = ap.parse_args()

    if args.only_split:
        recs = split_fasta(args.protein_file, args.tmp_dir)
        print(f"[split] {len(recs)} records -> {args.tmp_dir}")
        return
    if args.only_collect:
        n = collect(args.merged, set())
        print(f"[collect] wrote {n} preferred rank-0 Pareto sequences -> {args.merged}")
        return

    if not args.protein_file:
        ap.error("--protein-file required (unless --only-split/--only-collect)")

    recs = split_fasta(args.protein_file, args.tmp_dir)
    # resume: skip pids already in merged output
    seen = set()
    if os.path.exists(args.merged):
        with open(args.merged) as f:
            for line in f:
                if line.startswith(">"):
                    m = re.search(r"(NP_\d+\.\d+|XP_\d+\.\d+)", line)
                    if m:
                        seen.add(m.group(1))
    print(f"[batch] {len(recs)} records; {len(seen)} already done; "
          f"{len(recs) - len(seen)} to run")

    done = 0
    for i, (pid, rec_file) in enumerate(recs):
        short = pid.split(".")[0]
        if short in seen or pid in seen:
            continue
        print(f"[{i+1}/{len(recs)}] {pid} ...", flush=True)
        try:
            run_one(rec_file, args.host, args.gc_min, args.gc_max, pid)
        except RuntimeError as e:
            print(f"  FAILED {pid}: {e}", flush=True)
            continue
        done += 1
        seen.add(pid)

    n = collect(args.merged, set())
    print(f"[batch] done. ran {done} new; merged {n} preferred rank-0 Pareto sequences -> {args.merged}")


if __name__ == "__main__":
    main()
