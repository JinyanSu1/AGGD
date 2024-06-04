
seed=$1
num_cand=$2
num_iter=$3
k=$4
num_adv_passage_tokens=$5


method=$6
EVAL_MODEL=$7
EVAL_DATASET=$8
ATTK_MODEL=$9
ATTK_DATASET=${10}

python evaluate_adv.py \
   --attack_model_code ${ATTK_MODEL} --attack_dataset ${ATTK_DATASET} \
   --advp_path results --k ${k} \
   --eval_model_code ${EVAL_MODEL} --eval_dataset ${EVAL_DATASET} \
   --beir_results_path beir_result \
   --eval_res_path eval_result \
   --num_cand ${num_cand} \
   --num_iter ${num_iter} \
   --method ${method} \
   --num_adv_passage_tokens ${num_adv_passage_tokens} \
   --seed ${seed}

