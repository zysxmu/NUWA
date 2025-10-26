# Offical code for Large mRNA language foundation modeling with NUWA for unified sequence perception and generation


## Pretraining

1. Prepare the dataset

Please download the [data](https://www.ncbi.nlm.nih.gov/datasets/genome) and modify the 5, 6, and 7 lines in process_fna_taxonomy.py:

```
input_dir = "/path/to/your/data"
output_file = "/path/to/your/output_step1.txt"
data_summary_file = "/path/to/your/filtered_bacteria_species_updated_final.csv"

```
Then, run this command:

```
python process_fna_taxonomy.py

```

Run this command:

```
python process_data_memEffi.py "/path/to/your/output_step1.txt" "/path/to/your/output_step2.json"

```

Modify the 6 and 7 lines in process_data_class2rna_memEffi.py:

```
input_jsonl_path = "/path/to/your/output_step2.json"
output_json_path = "/path/to/your/output_step3.json"
```
Then, run this command:

```
python process_data_class2rna_memEffi.py
```

2. Run the pretraining command like the following example:

```
torchrun --nproc_per_node=4 pretrain.py \
 --train_data_path /path/to/your/output_step3.json \
 --model_dir /path/to/your/pretrained_model_checkpoint \
 --model_config_name "base" \
 --num_organisms 19676 \ # 19676 for bacteria, 4688 for eukaryote, 702 for archaea
 --max_length 1024 \ # 1024 for bacteria/archaea, 512 for eukaryote
 --num_class_per_batch 64 \
 --num_sample_per_class 4 \
 --num_class_per_batch_val 64 \
 --num_sample_per_class_val 4 \
 --learning_rate 0.00001 \
 --warmup_steps 10000 \
 --supcon_weight 1.0 \
 --mlm_prob_start 0.15 \
 --mlm_prob_end 0.5 \
 --log_steps 20 \
 --eval_steps_factor 1 \
 --save_steps_factor 0.1 \
 --num_train_epochs 10 \
 --val_ratio 0.2 \
 --test_ratio 0.1 \
 --eval_samples 2000
```

## The checkpoints

We have provided the pretrained model in [checkpoint](https://drive.google.com/drive/folders/1dK33csJUkqmhdL4hfpDoFm6AQXozcuo0?usp=drive_link)

## Finetuning (Perceptual task)

To finetune NUWA to compare with CodonBert, please follow these steps:

1. Prepare the dataset

Please download the dataset from [here](https://github.com/Sanofi-Public/CodonBERT)

2. For specific tasks, run the command like the following example:


   - Regression task

 ```
 CUDA_VISIBLE_DEVICES=0 python finetune_Reg.py \
 --pretrain_model_path "/path/to/your/pretrained_model_checkpoint" \ # here, we adopt the NUWA-archaea/NUWA-bacteria model
 --train_data_path "/path/to/your/regression_data.csv" \ # such as Fungal_expression.csv
 --checkpoint_dir "./finetune_regression_output" \
 --max_length 1024 \
 --species_id 0 \ # please specify the species id according to the dataset, and if there is no species, use the default value (0)
 --batch_size 8 \
 --learning_rate 5e-5 \
 --max_epochs 10 \
 --max_steps 1000 \
 --gradient_accumulation_steps 4 \
 --log_steps 10 \
 --eval_steps 10 \
 --save_steps 10 
 ```

 Note for mRNA stability tasks, we use the same setting as in CodonBert:

 ```
 python finetune_Reg_stability.py \
 --pretrain "/path/to/your/pretrained_model_checkpoint" \ # here, we adopt the NUWA-eukaryote model
 --model_dir "./stability-finetune-lora-output" \
 --lr 5e-5 \
 --batch 64 \
 --epochs 50 \
 --max_length 512
 ```

   - Classification task

 ```
 python finetune_Cls.py \
 --pretrain_model_path "/path/to/your/pretrained_model_checkpoint" \ # here, we adopt the NUWA-bacteria model
 --train_data_path "/path/to/your/E.Coli_proteins_data.csv" \
 --checkpoint_dir "./finetune_classification_output" \
 --max_length 1024 \
 --species_id 5834 \ # in NUWA-bacteria, this is the species id of E.Coli
 --batch_size 8 \
 --eval_batch_size_multiplier 100 \
 --learning_rate 5e-5 \
 --max_epochs 10 \
 --max_steps 1000 \
 --warmup_ratio 0.1 \
 --gradient_accumulation_steps 1 \
 --log_steps 10 \
 --eval_steps 10 \
 --save_strategy "no" \
 --save_total_limit 3 

 ```


3. To finetune NUWA to compare with BEACON, please unzip RNABenchmark-main.zip and follow it READMD.md

4. To finetune NUWA on protein, please use the zip in [here](https://drive.google.com/file/d/1SKWNNBqhU1CVcSuhk0GdVldeWLC_7JFp/view?usp=sharing)

## Generation (Generation task)

  - Generate mRNA for given Proteins

 ```
 python entropy_guide_mRNA_generation.py \
 --model_path "/path/to/your/pretrained_model_checkpoint" \
 --output_file "/path/to/your/output.txt" \
 --mode "protein_vectorized" \
 --protein_seq "your_protein_seqs" \
 --batch_size 64 \
 --temperature 1.0 \
 --top_p 1 \
 --class_id 5834 \ # # in NUWA-bacteria, this is the species id of E.Coli
 --model_max_length 1024  # Adjust if your model was trained with 1024
 ```

  - Generate Fixed-Length mRNA
 ```
 python entropy_guide_mRNA_generation.py \
 --model_path "/path/to/your/pretrained_model_checkpoint" \
 --output_file "/path/to/your/output.txt" \
 --mode "noprotein_vectorized" \
 --num_sequences 1000 \
 --seq_len 512 \
 --batch_size 64 \
 --temperature 1.0 \
 --top_p 1 \
 --class_id 5834 \ # # in NUWA-bacteria, this is the species id of E.Coli
 --model_max_length 512
 ```



## Acknowledgment

This code is based on CodonTransformer and CodonBert, and we appreciate their excellent works! The citations of CodonTransformer and CodonBert are provided as follows:

```sh
@article{Fallahpour_Gureghian_Filion_Lindner_Pandi_2025,
  title={CodonTransformer: a multispecies codon optimizer using context-aware neural networks},
  volume={16},
  ISSN={2041-1723},
  url={https://www.nature.com/articles/s41467-025-58588-7},
  DOI={10.1038/s41467-025-58588-7},
  number={1},
  journal={Nature Communications},
  author={Fallahpour, Adibvafa and Gureghian, Vincent and Filion, Guillaume J. and Lindner, Ariel B. and Pandi, Amir},
  year={2025},
  month=apr,
  pages={3205},
  language={en}
}

@article {Li2023.09.09.556981,
  author = {Sizhen Li and Saeed Moayedpour and Ruijiang Li and Michael Bailey and Saleh Riahi and Milad Miladi and Jacob Miner and Dinghai Zheng and Jun Wang and Akshay Balsubramani and Khang Tran and Minnie Zacharia and Monica Wu and Xiaobo Gu and Ryan Clinton and Carla Asquith and Joseph Skalesk and Lianne Boeglin and Sudha Chivukula and Anusha Dias and Fernando Ulloa Montoya and Vikram Agarwal and Ziv Bar-Joseph and Sven Jager},
  title = {CodonBERT: Large Language Models for mRNA design and optimization},
  elocation-id = {2023.09.09.556981},
  year = {2023},
  doi = {10.1101/2023.09.09.556981},
  publisher = {Cold Spring Harbor Laboratory},
  URL = {https://www.biorxiv.org/content/early/2023/09/12/2023.09.09.556981},
  eprint = {https://www.biorxiv.org/content/early/2023/09/12/2023.09.09.556981.full.pdf},
  journal = {bioRxiv}
}

```

## Issues

Please feel free to contact us (GitHub issues) for any questions about this code. 
