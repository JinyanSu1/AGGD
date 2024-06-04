 done


for dataset in nq-train msmarco; do
for seed in 1 2 3 4 5; do
for model in contriever contriever-msmarco dpr-single dpr-multi ance; do


bash ./scripts/AGGD.sh ${seed} 30 500 1 30 ${model} ${dataset}



done
done
done

