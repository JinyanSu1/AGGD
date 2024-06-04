



for dataset in msmarco; do

for model in contriever contriever-msmarco dpr-single dpr-multi ance; do

for seed in 1 2 3 4 5; do

for method in AGGD; do

bash ./scripts/eval_adv.sh ${seed} 30 2000 1 30 ${method} ${model} ${dataset} ${model} ${dataset}

done

done

done

done

for seed in 1 2 3 4 5; do
for method in AGGD; do
for model in contriever contriever-msmarco dpr-single dpr-multi ance; do

bash ./scripts/eval_adv.sh ${seed} 30 500 1 30 ${method} ${model} nq  ${model} nq-train
done
done
done