OUTPUT_PATH=results
Method=AGGD
seed=$1
num_cand=$2
num_iter=$3
k=$4
num_adv_passage_tokens=$5
MODEL=$6
DATASET=$7

for s in $(eval echo "{0..$((k-1))}"); do
output_file=${OUTPUT_PATH}/${Method}/${DATASET}/${MODEL}/k${k}-s${s}-seed${seed}-num_cand${num_cand}-num_iter${num_iter}-tokens${num_adv_passage_tokens}.json

python main.py \
--dataset ${DATASET} --split train \
--model_code ${MODEL} \
--num_cand ${num_cand} --per_gpu_eval_batch_size 64 --num_iter ${num_iter} --num_grad_iter 1 \
--output_file ${output_file} \
--do_kmeans --k $k --kmeans_split $s --seed $seed  --num_adv_passage_tokens $num_adv_passage_tokens 

done


