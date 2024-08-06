python train.py --task finetune --dataset_dir data --model_name sentence-transformers/LaBSE
python train.py --task predict --dataset_dir data --model_name semrel_baselines/models/finetuned_esp_labse
