import os
import csv

# 设置路径
input_dir = "/path/to/your/data"
output_file = "/path/to/your/output.txt"
data_summary_file = "/path/to/your/filtered_bacteria_species_updated_final.csv"

# 加载分类信息
taxonomy_info = {}
with open(data_summary_file, "r") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        gcf_id = row["Accession"].strip()
        tax_id = row.get("Tax_ID", "").strip()
        superkingdom = row.get("superkingdom", "").strip()
        kingdom = row.get("kingdom", "").strip()
        phylum = row.get("phylum", "").strip()
        class_ = row.get("class", "").strip()
        order = row.get("order", "").strip()
        family = row.get("family", "").strip()
        genus = row.get("genus", "").strip()
        species = row.get("species", "").strip()
        strain = row.get("Strain", "").strip()

        taxonomy_info[gcf_id] = {
            "Tax_ID": tax_id,
            "superkingdom": superkingdom,
            "kingdom": kingdom,
            "phylum": phylum,
            "class": class_,
            "order": order,
            "family": family,
            "genus": genus,
            "species": species,
            "strain": strain
        }

# 处理所有 fna 文件
with open(output_file, "w") as out_f:
    for accession in taxonomy_info.keys():
        # 查找匹配的 fna 文件
        matched_file = None
        for root, _, files in os.walk(input_dir):
            for file_name in files:
                if file_name.startswith(accession) and file_name.endswith("cds_from_genomic.fna"):
                    matched_file = os.path.join(root, file_name)
                    break
            if matched_file:
                break

        # 如果找到匹配的文件，则处理该文件
        if matched_file:
            tax_info = taxonomy_info[accession]
            tax_id = tax_info.get("Tax_ID", "")
            superkingdom = tax_info.get("superkingdom", "")
            kingdom = tax_info.get("kingdom", "")
            phylum = tax_info.get("phylum", "")
            class_ = tax_info.get("class", "")
            order = tax_info.get("order", "")
            family = tax_info.get("family", "")
            genus = tax_info.get("genus", "")
            species = tax_info.get("species", "")
            strain = tax_info.get("strain", "")

            # 读取 fna 文件并提取序列
            with open(matched_file, "r") as fna_file:
                sequence = ""
                for line in fna_file:
                    if line.startswith(">"):
                        if sequence:
                            rna_sequence = sequence.replace("T", "U")
                            out_f.write(f"{rna_sequence},{tax_id},{superkingdom},{kingdom},{phylum},{class_},{order},{family},{genus},{species},{strain}\n")
                            sequence = ""
                    else:
                        sequence += line.strip()
                # 写入最后一个序列
                if sequence:
                    rna_sequence = sequence.replace("T", "U")
                    out_f.write(f"{rna_sequence},{tax_id},{superkingdom},{kingdom},{phylum},{class_},{order},{family},{genus},{species},{strain}\n")
        else:
            print(f"警告: 找不到与 {accession} 匹配的 fna 文件，跳过此条目。")

    print(f"Done processing and saved to {output_file}")

