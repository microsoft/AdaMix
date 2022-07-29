. ./venv/bin/activate

seed=110

python -m torch.distributed.launch --nproc_per_node=1 src/gpt2_ft.py \
--train_data ./data/e2e/train.jsonl \
--valid_data ./data/e2e/valid.jsonl \
--train_batch_size 8 \
--grad_acc 1 \
--valid_batch_size 4 \
--seq_len 512 \
--model_card gpt2.md \
--init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin \
--platform local \
--clip 0.0 \
--lr 0.0002 \
--weight_decay 0.01 \
--correct_bias \
--adam_beta2 0.999 \
--scheduler linear \
--warmup_step 500 \
--max_epoch 5 \
--save_interval 1000 \
--lora_dim 4 \
--lora_alpha 32 \
--lora_dropout 0.1 \
--label_smooth 0.1 \
--work_dir ./trained_models/GPT2_M/e2e/$seed/lora_only \
--random_seed $seed \
--lora_only 1

bash run_eval_e2e_lora_only.sh --seed $seed