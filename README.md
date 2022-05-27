# AdaMix (Mixture-of-Adapter)


This is the implementation of the paper [AdaMix: Mixture-of-Adapter for Parameter-efficient Tuning of Large Language Models](https://arxiv.org/abs/2205.12410). 

## Adapting to the GLUE Benchmark
Our experiments on the GLUE benchmark are run on 16 NVIDIA Tesla V100 GPU. The results may vary due to different GPU models, drivers, CUDA SDK versions, floating-point precisions, and random seeds. 


## Download AdaMix checkpoints
We release all copies of Adapter weights for users' Adapter aggregation study. 

|   | Dataset  | BERT base 110M <br>   | RoBERTa large 355M <br>  |
|---|----------|--------------------|----------------------|
|   | MNLI     |[8.5 MB](https://github.com/yaqingwang/MoA/releases/download/bert_base/pytorch_model_mnli_expert_soup.bin) |[11.7 MB](https://github.com/yaqingwang/MoA/releases/download/roberta_large/pytorch_model_mnli_expert_soup.bin) |
|   | SST2     |[8.5 MB](https://github.com/yaqingwang/MoA/releases/download/bert_base/pytorch_model_sst2_expert_soup.bin)  |[11.7 MB](https://github.com/yaqingwang/MoA/releases/download/roberta_large/pytorch_model_sst2_expert_soup.bin)  |
|   | MRPC     |[8.5 MB](https://github.com/yaqingwang/MoA/releases/download/bert_base/pytorch_model_mrpc_expert_soup.bin)  |[11.7 MB](https://github.com/yaqingwang/MoA/releases/download/roberta_large/pytorch_model_mrpc_expert_soup.bin)  |
|   | CoLA     |[8.5 MB](https://github.com/yaqingwang/MoA/releases/download/bert_base/pytorch_model_cola_expert_soup.bin)  |[11.7 MB](https://github.com/yaqingwang/MoA/releases/download/roberta_large/pytorch_model_cola_expert_soup.bin)  |
|   | QNLI     |[8.5 MB](https://github.com/yaqingwang/MoA/releases/download/bert_base/pytorch_model_qnli_expert_soup.bin)  |[11.7 MB](https://github.com/yaqingwang/MoA/releases/download/roberta_large/pytorch_model_qnli_expert_soup.bin)  |
|   | QQP      |[8.5 MB](https://github.com/yaqingwang/MoA/releases/download/bert_base/pytorch_model_qqp_expert_soup.bin)  |[11.7 MB](https://github.com/yaqingwang/MoA/releases/download/roberta_large/pytorch_model_qqp_expert_soup.bin)  |
|   | RTE      |[8.5 MB](https://github.com/yaqingwang/MoA/releases/download/bert_base/pytorch_model_rte_expert_soup.bin)  |[11.7 MB](https://github.com/yaqingwang/MoA/releases/download/roberta_large/pytorch_model_rte_expert_soup.bin)  |
|   | STSB     |[8.5 MB](https://github.com/yaqingwang/MoA/releases/download/bert_base/pytorch_model_stsb_expert_soup.bin)  |[11.7 MB](https://github.com/yaqingwang/MoA/releases/download/roberta_large/pytorch_model_stsb_expert_soup.bin)  |

## Steps to reproduce our results
### Create and activate conda env
```console
conda env create -f environment.yml
```
### Install the pre-requisites
```console
pip install -e .
```

We also provide the shell scripts for bert-base and roberta-large.

### Quick start
```console
export num_gpus=1
export PYTHONHASHSEED=0
task_name=mnli
model=roberta-large
export output_dir="./models/${model}/${task_name}"
python -m torch.distributed.launch --nproc_per_node=$num_gpus \
examples/text-classification/run_glue.py \
--model_name_or_path $model \
--task_name $task_name \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 32 \
--learning_rate 3e-4 \
--num_train_epochs 20 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 1000 \
--logging_dir $output_dir/log \
--evaluation_strategy epoch \
--save_strategy epoch \
--warmup_ratio 0.06 \
--apply_expert_soup \
--adapter_size 16 \
--num_experts 4 \
--seed 0 \
--inference_level 3 \
--weight_decay 0.1 \
--sharing_up 1 \
--sharing_down 0 \
--use_consistency_loss 1

```
Most arguments are inherited from transformers and are easy to understand. We further explain some of the AdaMix's arguments:
* `inference_level`: There are two suggested modes
  * `1`: Random Routing
  * `3`: Averaging the weights of Adapters for routing (used in AdaMix)

* `num_experts`: Number of Adapters in AdaMix

* `use_consistency_loss`: Two modes. 
  * `0`: No consistency loss
  * `1`: Use consistency loss


* `sharing_up`: There are two modes. (sharing_down is same)
  * `0`: No weight sharing
  * `1`: Sharing Project-up layer weights in Adapter



### Evaluate the checkpoints
Create checkpoints directory and download checkpoints of corresponding tasks under the directory. Use MNLI as an example. Use your checkpoint path in **expert_soup_path** argument.
```console
export num_gpus=1
export PYTHONHASHSEED=0
task_name=mnli
model=roberta-large
export output_dir="./models/${model}/${task_name}"
python -m torch.distributed.launch --nproc_per_node=$num_gpus \
examples/text-classification/run_glue.py \
--model_name_or_path $model \
--task_name $task_name \
--do_eval \
--expert_soup_path ./checkpoints/pytorch_model_${task_name}_expert_soup.bin \
--max_seq_length 128 \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 32 \
--learning_rate 3e-4 \
--num_train_epochs 20 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 1000 \
--logging_dir $output_dir/log \
--evaluation_strategy epoch \
--save_strategy epoch \
--warmup_ratio 0.06 \
--apply_expert_soup \
--adapter_size 16 \
--num_experts 4 \
--seed 0 \
--inference_level 3 \
--weight_decay 0.1 \
--sharing_up 1 \
--sharing_down 0 \
--use_consistency_loss 1

```

### Notes and Acknowledgments
The implementation is based on https://github.com/huggingface/transformers  <br>
We also used some code from: https://github.com/microsoft/LoRA 


### How do I cite AdaMix?

```
@article{wang2022adamix,
  title={AdaMix: Mixture-of-Adapter for Parameter-efficient Tuning of Large Language Models},
  author={Wang, Yaqing and Mukherjee, Subhabrata and Liu, Xiaodong and Gao, Jing and Awadallah, Ahmed Hassan and Gao, Jianfeng},
  journal={arXiv preprint arXiv:2205.12410},
  year={2022}
}
```