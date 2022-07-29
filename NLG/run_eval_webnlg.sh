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
    --data ./data/webnlg_challenge_2017/test.jsonl \
    --batch_size 1 \
    --seq_len 256 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M/webnlg/$seed/$vv/model.final.pt \
    --platform local \
    --lora_dim 4 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M/webnlg/$seed/$vv \
    --output_file predict.jsonl \
    --n_experts $n_experts \
    --share_A 0 \
    --share_B 1

python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_M/webnlg/$seed/$vv/predict.jsonl \
    --input_file ./data/webnlg_challenge_2017/test_formatted.jsonl \
    --ref_type webnlg \
    --ref_num 6 \
    --output_ref_file eval/GenerationEval/data/references_webnlg \
    --output_pred_file eval/GenerationEval/data/hypothesis_webnlg \
    --tokenize --lower

cd ./eval/GenerationEval/
python eval.py \
    -R data/references_webnlg/reference \
    -H data/hypothesis_webnlg \
    -nr 6 \
    -m bleu,meteor,ter 
cd ../..