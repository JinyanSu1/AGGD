

for dataset in nq msmarco; do

for model in contriever contriever-msmarco dpr-single dpr-multi ance; do

FILE=beir_result


python evaluation_beir.py --model_code $model --dataset $dataset --result_output $FILE 


done

done

