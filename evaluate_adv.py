from beir import util
from beir.datasets.data_loader import GenericDataLoader
import os
import json
import sys
import argparse
import torch
import copy
from utils.load_model import load_models
sys.path.append("./contriever/src")
sys.path.append("./contriever")
from tqdm import tqdm



def evaluate_recall(results, qrels, k_values = [50,1000]):
    cnt = {k: 0 for k in k_values}
    for q in results:
        sims = list(results[q].items())
        sims.sort(key=lambda x: x[1], reverse=True)
        gt = qrels[q]
        found = 0
        for i, (c, _) in enumerate(sims[:max(k_values)]):
            if c in gt:
                found = 1
            if (i + 1) in k_values:
                cnt[i + 1] += found
#             print(i, c, found)
    recall = {}
    for k in k_values:
        recall[f"Recall@{k}"] = round(cnt[k] / len(results), 5)
    
    return recall


def main():
    parser = argparse.ArgumentParser(description='test')
    # The model and dataset used to generate adversarial passages 
    parser.add_argument("--attack_model_code", type=str, default="contriever", choices=["contriever-msmarco", "contriever", "dpr-single", "dpr-multi", "ance"])
    parser.add_argument("--attack_dataset", type=str, default="nq-train", choices=["nq-train", "msmarco"])
    parser.add_argument("--advp_path", type=str, default="results", help="the path where generated adversarial passages are stored")
    parser.add_argument("--k", type=int, default=1, help="how many adversarial passages are generated (i.e., k in k-means); you may test multiple by passing")

    # The model and dataset used to evaluate the attack performance (e.g., if eval_model is different from attack_model, it studies attack across different models)
    parser.add_argument("--eval_model_code", type=str, default="contriever", choices=["contriever-msmarco", "contriever", "dpr-single", "dpr-multi", "ance"])
    parser.add_argument('--eval_dataset', type=str, default="fiqa", help='BEIR dataset to evaluate')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument("--beir_results_path", type=str, default='beir_result', help='Eval results path of eval_model on the beir eval_dataset')

    # Where to save the evaluation results (attack performance)

    parser.add_argument("--eval_res_path", type=str, default="eval_result")

    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--pad_to_max_length', default=True)
    parser.add_argument('--method', type = str, default = 'AGGD')
    parser.add_argument('--seed', type = int, default = 10)
    parser.add_argument("--num_cand", default=150, type=int)
    parser.add_argument("--num_iter", default=2000, type=int)
    parser.add_argument("--num_adv_passage_tokens", default=30, type=int)




    args = parser.parse_args()
    sub_dir = '%s/%s/%s/%s'%(args.eval_res_path, args.method, args.attack_dataset, args.attack_model_code)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir, exist_ok=True)

    filename = '%s/%s-%s-k%d-seed%d-num_cand%d-num_iter%d-tokens%d.json'%(sub_dir, args.eval_dataset, args.eval_model_code, args.k, args.seed, args.num_cand, args.num_iter, args.num_adv_passage_tokens) 
    if os.path.isfile(filename):
        return
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.eval_dataset)
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = os.path.join(out_dir, args.eval_dataset)
    if not os.path.exists(data_path):
        data_path = util.download_and_unzip(url, out_dir)
    print(data_path)

    data = GenericDataLoader(data_path)
    if '-train' in data_path:
        args.split = 'train'
    corpus, queries, qrels = data.load(split=args.split)

    
    beir_result_file = f'{args.beir_results_path}/{args.eval_dataset}/{args.eval_model_code}/beir.json'
    with open(beir_result_file, 'r') as f:
        results = json.load(f)
    
    assert len(qrels) == len(results)
    print('Total samples:', len(results))

    # Load models
    model, c_model, tokenizer,get_emb = load_models(args.eval_model_code)

    model.eval()
    model.cuda()
    c_model.eval()
    c_model.cuda()

    def evaluate_adv(k, qrels, results):
        adv_ps = []
        for s in range(k):
            file_name = "%s/%s/%s/%s/k%d-s%d-seed%d-num_cand%d-num_iter%d-tokens%d.json"%(args.advp_path, args.method, args.attack_dataset, args.attack_model_code, k, s, args.seed, args.num_cand, args.num_iter, args.num_adv_passage_tokens)
            if not os.path.exists(file_name):
                print(f"!!!!! {file_name} does not exist!")
                continue
            attack_results = []
            
            with open(file_name, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    attack_results.append(data)
                
                adv_ps.append(attack_results[-1])
        print('# adversaria passages', len(adv_ps))
        
        
        adv_results = copy.deepcopy(results)
        
        adv_p_ids = [tokenizer.convert_tokens_to_ids(p["best_adv_text"]) for p in adv_ps]
        adv_p_ids = torch.tensor(adv_p_ids).cuda()
        adv_attention = torch.ones_like(adv_p_ids, device='cuda')
        adv_token_type = torch.zeros_like(adv_p_ids, device='cuda')
        adv_input = {'input_ids': adv_p_ids, 'attention_mask': adv_attention, 'token_type_ids': adv_token_type}
        with torch.no_grad():
            adv_embs = get_emb(c_model, adv_input)
        
        adv_qrels = {q: {"adv%d"%(s):1 for s in range(k)} for q in qrels}
        
        for i, query_id in tqdm(enumerate(results)):
            query_text = queries[query_id]
            query_input = tokenizer(query_text, padding=True, truncation=True, return_tensors="pt")
            query_input = {key: value.cuda() for key, value in query_input.items()}
            with torch.no_grad():
                query_emb = get_emb(model, query_input)
                adv_sim = torch.mm(query_emb, adv_embs.T)
            
            for s in range(len(adv_ps)):
                adv_results[query_id]["adv%d"%(s)] = adv_sim[0][s].cpu().item()
        
        adv_eval = evaluate_recall(adv_results, adv_qrels)


        return adv_eval
    


    
    final_res = evaluate_adv(args.k, qrels, results)
    



    
    with open(filename, 'w') as f:
        json.dump(final_res, f)

if __name__ == "__main__":
    main()