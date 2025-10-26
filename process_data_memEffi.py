import json
from tqdm import tqdm
import os

def process_rna_data_with_progress(input_txt_path, output_json_path):
    species_set = set()

    # Get total number of lines for progress bar
    with open(input_txt_path, 'r') as f:
        total_lines = sum(1 for _ in f)

    print("🔍 Scanning all species...")
    with open(input_txt_path, 'r') as f:
        for line in tqdm(f, total=total_lines, desc="Scanning species"):
            parts = line.strip().split(',')
            if len(parts) >= 9:
                species_name = parts[-2].strip()
                if species_name:
                    species_set.add(species_name)

    CLASS_TO_ID = {species: idx for idx, species in enumerate(sorted(species_set))}
    print(f"✅ Found {len(CLASS_TO_ID)} unique species")

    print("\n🔄 Converting data and writing to file...")
    skipped_lines = 0
    unknown_species_count = 0
    total_sequences = 0

    with open(input_txt_path, 'r') as fin, open(output_json_path, 'w') as fout:
        for line in tqdm(fin, total=total_lines, desc="Converting"):
            parts = line.strip().split(',')
            if len(parts) < 9:
                skipped_lines += 1
                continue

            rna_seq = parts[0].strip()
            species_name = parts[-2].strip()

            codons = ' '.join([rna_seq[i:i+3] for i in range(0, len(rna_seq), 3)])
            organism_id = CLASS_TO_ID.get(species_name, -1)

            if organism_id == -1:
                unknown_species_count += 1

            record = {
                "codons": codons,
                "organism": organism_id
            }

            fout.write(json.dumps(record) + '\n')
            total_sequences += 1

    # Save species mapping file
    with open('species_mapping.json', 'w') as f:
        json.dump({
            "species_to_id": CLASS_TO_ID,
            "id_to_species": {v: k for k, v in CLASS_TO_ID.items()}
        }, f, indent=2)

    # Summary information
    print(f"\n🎉 Conversion completed!")
    print(f"• Input file: {os.path.abspath(input_txt_path)}")
    print(f"• Output file: {os.path.abspath(output_json_path)}")
    print(f"• Species mapping: {os.path.abspath('species_mapping.json')}")
    print(f"• Total sequences: {total_sequences:,}")
    print(f"• Skipped incomplete lines: {skipped_lines:,}")
    print(f"• Unknown species sequences: {unknown_species_count:,}")


# 使用示例
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    process_rna_data_with_progress(args.input, args.output)