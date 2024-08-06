#!/bin/bash

# List of languages
# languages=("amh" "arb" "arq" "ary" "esp" "hau" "hin" "ind" "kin" "mar" "pan" "tel" "afr")
# languages=("tel" "amh")
# languages=("pan" "ind" "hau")
languages=("eng")
# Loop through the languages
for lang in "${languages[@]}"
do
    echo "Running evaluate_sbert.py for $lang"
    python -m src.evaluate_sbert --dataset_dir data/$lang --model_name models/finetuned_esp_labse
done