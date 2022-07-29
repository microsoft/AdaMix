. ./venv/bin/activate
sudo apt install default-jre -y

seed=110
n_experts=8
vv=lora_adamix

while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    --seed)
    seed=$2
    shift
    shift
    ;;
    --n_experts)
    n_experts=$2
    shift
    shift
    ;;
    --vv)
    vv=$2
    shift
    shift
    ;;
esac
done

python -m torch.distributed.launch --nproc_per_node=16 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 128 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M/e2e/$seed/$vv/model.final.pt \
    --platform local \
    --lora_dim 4 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M/e2e/$seed/$vv \
    --output_file predict.jsonl \
    --n_experts $n_experts \
    --share_A 0 \
    --share_B 1

python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_M/e2e/$seed/$vv/predict.jsonl \
    --input_file ./data/e2e/test_formatted.jsonl \
    --output_ref_file e2e_ref.txt \
    --output_pred_file e2e_pred.txt

python eval/e2e/measure_scores.py e2e_ref.txt e2e_pred.txt -p