import json
import os
from tqdm import tqdm
import shutil

input_jsonl_path = "/path/to/your/output_step2.json"
output_json_path = "/path/to/your/output_step3.json"
tmp_dir = "tmp_class_data"

# Create temporary directory
os.makedirs(tmp_dir, exist_ok=True)

# Step 1: Read JSON Lines line by line, write to temp files by class (low memory)
print("📂 Step 1: Writing species to temporary files (Streaming)")
with open(input_jsonl_path, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in open(input_jsonl_path, "r", encoding="utf-8"))
    for i, line in enumerate(tqdm(f, total=total_lines, desc="Sorting", ncols=100)):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            organism = str(item["organism"])
            codons = item["codons"].strip()
            if codons:
                tmp_file_path = os.path.join(tmp_dir, f"{organism}.txt")
                with open(tmp_file_path, "a", encoding="utf-8") as tmp_f:
                    tmp_f.write(codons + '\n')
        except Exception as e:
            tqdm.write(f"❌ Error parsing line {i}: {e}")

# Step 2: Read each species temp file, and stream write to final JSON file (avoid building large dict)
print("\n💾 Step 2: Writing final JSON file (Streaming)")
with open(output_json_path, "w", encoding="utf-8") as fout:
    fout.write('{\n')
    first = True
    total_classes = 0
    total_samples = 0

    for fname in tqdm(os.listdir(tmp_dir), desc="Writing", ncols=100):
        organism = os.path.splitext(fname)[0]
        file_path = os.path.join(tmp_dir, fname)

        with open(file_path, "r", encoding="utf-8") as f:
            codon_seqs = [line.strip() for line in f if line.strip()]

        import random
        # random deletion
        random_deletion = 0
        if len(codon_seqs) > 100:
            random.shuffle(codon_seqs)
            num_to_keep = int(len(codon_seqs) * (1 - random_deletion))
            codon_seqs = codon_seqs[:num_to_keep]


        if len(codon_seqs) >= 4:
            if not first:
                fout.write(',\n')
            json_line = json.dumps(organism) + ': ' + json.dumps(codon_seqs, ensure_ascii=False)
            fout.write(json_line)
            first = False
            total_classes += 1
            total_samples += len(codon_seqs)

        os.remove(file_path)

    fout.write('\n}\n')

# Step 3: Clean up temporary directory
# shutil.rmtree(tmp_dir)

# ✅ Summary output
print(f"\n✅ Result saved: {output_json_path}")
print(f"- Total classes: {total_classes}")
print(f"- Total samples: {total_samples}")