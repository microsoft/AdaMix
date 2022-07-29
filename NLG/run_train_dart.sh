. ./venv/bin/activate

seed=110
n_experts=8

python -m torch.distributed.launch --nproc_per_node=1 src/gpt2_ft.py \
--train_data ./data/dart/train.jsonl \
--valid_data ./data/dart/valid.jsonl \
--train_batch_size 8 \
--grad_acc 1 \
--valid_batch_size 4 \
--seq_len 512 \
--model_card gpt2.md \
--init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin \
--platform local \
--clip 0.0 \
--lr 0.0002 \
--weight_decay 0.0 \
--correct_bias \
--adam_beta2 0.999 \
--scheduler linear \
--warmup_step 2000 \
--max_epoch 20 \
--eval_interval 5000 \
--save_interval 5000 \
--lora_dim 4 \
--lora_alpha 32 \
--lora_dropout 0.0 \
--label_smooth 0.0 \
--work_dir ./trained_models/GPT2_M/dart/$seed/lora_adamix \
--random_seed $seed \
--n_experts $n_experts \
--share_A 0 \
--share_B 1

bash run_eval_dart.sh --seed $seed --n_experts $n_experts