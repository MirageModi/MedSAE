#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import json
from collections import defaultdict
from tqdm import tqdm
import os
from scipy import stats

SENSE_TO_NER_MAP = {
    "administrative": ["administrative healthcare event", "medical procedure or intervention"],
    "clinical": ["bodily fluid or secretion", "clinical finding or symptom"],
    "radiological": ["radiological finding or shadow", "anatomical structure"],
    "anatomical": ["anatomical structure"],
    "dermatological": ["dermatological or skin lesion"],
    "pathological": ["clinical finding or symptom", "dermatological or skin lesion", "radiological finding or shadow", "anatomical structure"],
    "cardiac": ["cardiovascular disorder", "medical procedure or intervention"],
    "cardiovascular": ["cardiovascular disorder", "medical procedure or intervention"],
    "GI": ["gastrointestinal disorder", "clinical finding or symptom"],
    "neurological": ["neurological or mental disorder", "medical procedure or intervention"],
    "psychological": ["neurological or mental disorder"],
    "respiratory": ["respiratory or pulmonary disorder", "medical procedure or intervention"],
    "orthopedic": ["musculoskeletal or orthopedic disorder", "medical procedure or intervention"],
    "renal": ["renal or kidney disorder", "laboratory measurement"],
    "endocrine": ["endocrine disorder", "laboratory measurement"],
    "surgical": ["medical procedure or intervention"],
    "therapeutic": ["medication or therapeutic agent"],
}

TARGET_WORDS = ["discharge", "failure", "block", "pressure", "lesion", "syndrome", "mass", "episode", "effusion", "arrest", "attack", "shock"]


def parse_latents(row):
    try:
        ids = json.loads(row['latent_ids_json'])
        vals = json.loads(row['z_values_json'])
        return sorted(zip(ids, vals), key=lambda x: x[1], reverse=True)
    except:
        return []


def get_feature_subsets(pairs, n_anchor=10):
    full_vec = {int(id_): float(val) for id_, val in pairs}
    anchor_vec = {int(id_): float(val) for id_, val in pairs[:n_anchor]}
    semantic_vec = {int(id_): float(val) for id_, val in pairs[n_anchor:]}
    return full_vec, anchor_vec, semantic_vec


def sparse_cosine_similarity(vec_a, vec_b):
    if not vec_a or not vec_b:
        return 0.0
    common = set(vec_a.keys()) & set(vec_b.keys())
    if not common:
        return 0.0
    dot = sum(vec_a[k] * vec_b[k] for k in common)
    norm_a = np.sqrt(sum(v**2 for v in vec_a.values()))
    norm_b = np.sqrt(sum(v**2 for v in vec_b.values()))
    return dot / (norm_a * norm_b + 1e-9)


def compute_retrieval_metrics(query_label, retrieved_labels, k_values=[1, 3, 5, 10]):
    metrics = {}
    for k in k_values:
        if k <= len(retrieved_labels):
            metrics[f'P@{k}'] = sum(1 for s in retrieved_labels[:k] if s == query_label) / k
        else:
            metrics[f'P@{k}'] = 0.0
    
    mrr = 0.0
    for i, label in enumerate(retrieved_labels):
        if label == query_label:
            mrr = 1.0 / (i + 1)
            break
    metrics['MRR'] = mrr
    
    relevant, precision_sum = 0, 0.0
    for i, label in enumerate(retrieved_labels):
        if label == query_label:
            relevant += 1
            precision_sum += relevant / (i + 1)
    total_relevant = sum(1 for s in retrieved_labels if s == query_label)
    metrics['AP'] = precision_sum / total_relevant if total_relevant > 0 else 0.0
    return metrics


def load_ner_by_char_offset(ner_path):
    ner_exact, ner_by_doc = {}, defaultdict(list)
    print(f"Loading NER spans from {ner_path}...")
    with open(ner_path) as f:
        for line in tqdm(f, desc="Loading NER"):
            if not line.strip():
                continue
            obj = json.loads(line)
            doc_id, grp = str(obj.get('doc_id', '')), obj.get('entity_group')
            start, end = int(obj.get('start', 0)), int(obj.get('end', 0))
            ner_exact[(doc_id, start)] = grp
            ner_by_doc[doc_id].append({'start': start, 'end': end, 'grp': grp})
    return ner_exact, ner_by_doc


def find_ner_label(doc_id, char_start, char_end, ner_exact, ner_by_doc):
    if (doc_id, char_start) in ner_exact:
        return ner_exact[(doc_id, char_start)]
    for span in ner_by_doc.get(doc_id, []):
        if span['end'] > char_start and span['start'] < char_end:
            return span['grp']
    return None


def resolve_sense(ner_label, valid_senses):
    if not ner_label:
        return None
    ner_label = ner_label.lower().strip()
    # Sort senses for deterministic ordering
    for sense in sorted(valid_senses):
        for expected in SENSE_TO_NER_MAP.get(sense, [sense]):
            if ner_label == expected.lower():
                return sense
    return None

def run_fine_grained_k_sweep(df_processed, word_to_senses, n_anchor=10):
    """Fine-grained k sweep around the optimal range."""
    print("\n" + "="*70)
    print("EXPERIMENT A: FINE-GRAINED K SWEEP FOR RE-RANKING")
    print("="*70)
    
    k_values = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
    results = []
    per_query_results = []
    
    for word in TARGET_WORDS:
        word_df = df_processed[df_processed['lemma_lower'] == word]
        if len(word_df) < 30:
            continue
        
        sense_counts = word_df['sense'].value_counts()
        valid_senses = sense_counts[sense_counts >= 5].index.tolist()
        if len(valid_senses) < 2:
            continue
        
        word_df = word_df[word_df['sense'].isin(valid_senses)]
        if len(word_df) < 20:
            continue
        
        print(f"Processing: {word} ({len(word_df)} samples, senses: {dict(word_df['sense'].value_counts())})")
        indices = word_df.index.tolist()
        
        for rerank_k in k_values:
            for query_idx in indices:
                query_row = word_df.loc[query_idx]
                query_sense = query_row['sense']
                
                candidates = []
                for cand_idx in indices:
                    if cand_idx == query_idx:
                        continue
                    cand_row = word_df.loc[cand_idx]
                candidates.append({
                    'sense': cand_row['sense'],
                    'full_sim': sparse_cosine_similarity(query_row['full_vec'], cand_row['full_vec']),
                    'semantic_sim': sparse_cosine_similarity(query_row['semantic_vec'], cand_row['semantic_vec'])
                })
                
                candidates_full = sorted(candidates, key=lambda x: x['full_sim'], reverse=True)
                metrics_full = compute_retrieval_metrics(query_sense, [c['sense'] for c in candidates_full])
                
                stage1 = sorted(candidates, key=lambda x: x['full_sim'], reverse=True)[:rerank_k]
                reranked = sorted(stage1, key=lambda x: x['semantic_sim'], reverse=True)
                remaining = [c for c in candidates_full if c not in stage1]
                metrics_rerank = compute_retrieval_metrics(query_sense, [c['sense'] for c in reranked + remaining])
                
                per_query_results.append({
                    'word': word, 'rerank_k': rerank_k, 'query_sense': query_sense,
                    'full_P@1': metrics_full['P@1'], 'rerank_P@1': metrics_rerank['P@1'],
                    'full_AP': metrics_full['AP'], 'rerank_AP': metrics_rerank['AP'],
                    'delta_P@1': metrics_rerank['P@1'] - metrics_full['P@1'],
                    'delta_AP': metrics_rerank['AP'] - metrics_full['AP']
                })
    
    per_query_df = pd.DataFrame(per_query_results)
    
    print("\n" + "-"*70)
    print("FINE-GRAINED K SWEEP RESULTS")
    print("-"*70)
    print(f"{'k':<6} | {'ΔP@1':>8} | {'95% CI':>15} | {'ΔMAP':>8} | {'p-value':>10}")
    print("-"*65)
    
    summary_rows = []
    for k in k_values:
        k_df = per_query_df[per_query_df['rerank_k'] == k]
        
        delta_p1 = k_df['delta_P@1'].values
        delta_ap = k_df['delta_AP'].values
        
        mean_p1 = np.mean(delta_p1) * 100
        std_p1 = np.std(delta_p1) * 100
        ci_p1 = 1.96 * std_p1 / np.sqrt(len(delta_p1))
        
        mean_ap = np.mean(delta_ap) * 100
        
        t_stat, p_value = stats.ttest_rel(k_df['rerank_P@1'], k_df['full_P@1'])
        
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        
        print(f"k={k:<4} | {mean_p1:>+7.2f}% | [{mean_p1-ci_p1:>+.2f}, {mean_p1+ci_p1:>+.2f}] | {mean_ap:>+7.2f}% | {p_value:>9.4f} {sig}")
        
        summary_rows.append({
            'k': k, 'delta_P@1_mean': mean_p1, 'delta_P@1_ci': ci_p1,
            'delta_MAP_mean': mean_ap, 'p_value': p_value, 'n': len(k_df)
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    optimal_k = summary_df.loc[summary_df['delta_P@1_mean'].idxmax(), 'k']
    print(f"\nOptimal k for P@1 improvement: k={optimal_k}")
    
    return per_query_df, summary_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk_csv", required=True)
    parser.add_argument("--ner_jsonl", required=True)
    parser.add_argument("--lexicon_csv", required=True)
    parser.add_argument("--output_dir", default="final_paper_results")
    parser.add_argument("--n_anchor", type=int, default=10)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading data...")
    df = pd.read_csv(args.topk_csv, low_memory=False)
    df['lemma_lower'] = df['lemma'].str.lower()
    df['doc_id'] = df['doc_id'].astype(str)
    
    df = df.dropna(subset=['char_start', 'char_end'])
    df['char_start'] = df['char_start'].astype(int)
    df['char_end'] = df['char_end'].astype(int)
    df = df[(df['char_start'] > 0) | (df['char_end'] > 0)]
    
    df_lex = pd.read_csv(args.lexicon_csv)
    word_to_senses = defaultdict(set)
    for _, row in df_lex.iterrows():
        word_to_senses[row['term'].lower()].add(row['sense'].lower())
    
    ner_exact, ner_by_doc = load_ner_by_char_offset(args.ner_jsonl)
    
    def get_sense(row):
        valid = word_to_senses.get(row['lemma_lower'], set())
        if not valid:
            return None
        ner = find_ner_label(str(row['doc_id']), int(row['char_start']), int(row['char_end']), ner_exact, ner_by_doc)
        return resolve_sense(ner, valid)
    
    df['sense'] = df.apply(get_sense, axis=1)
    df = df.dropna(subset=['sense'])
    
    df['pairs'] = df.apply(parse_latents, axis=1)
    df = df[df['pairs'].apply(len) > args.n_anchor]
    
    subsets = df['pairs'].apply(lambda p: get_feature_subsets(p, args.n_anchor))
    df['full_vec'] = subsets.apply(lambda x: x[0])
    df['anchor_vec'] = subsets.apply(lambda x: x[1])
    df['semantic_vec'] = subsets.apply(lambda x: x[2])
    
    print(f"Final processed rows: {len(df)}")
    
    exp_a_detail, exp_a_summary = run_fine_grained_k_sweep(df, word_to_senses, args.n_anchor)
    exp_a_detail.to_csv(f"{args.output_dir}/exp_a_k_sweep_detail.csv", index=False)
    exp_a_summary.to_csv(f"{args.output_dir}/exp_a_k_sweep_summary.csv", index=False)
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        fig, axes = plt.subplots(1, 1, figsize=(8, 3))
        
        ax1 = axes[0]
        k_vals = exp_a_summary['k'].values
        means = exp_a_summary['delta_P@1_mean'].values
        cis = exp_a_summary['delta_P@1_ci'].values
        
        ax1.errorbar(k_vals, means, yerr=cis, fmt='o-', capsize=4, capthick=2, 
                    linewidth=2, markersize=8, color='#d62728')
        ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax1.fill_between(k_vals, means-cis, means+cis, alpha=0.2, color='#d62728')
        
        optimal_idx = np.argmax(means)
        ax1.scatter([k_vals[optimal_idx]], [means[optimal_idx]], s=200, color='gold', 
                   edgecolor='black', zorder=5, marker='*')
        
        ax1.set_xlabel('Candidate Pool Size (k)', fontsize=12)
        ax1.set_ylabel('ΔP@1 (Re-rank − Baseline) %', fontsize=12)
        ax1.set_title('A. Re-ranking Improvement\nby Candidate Pool Size', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/figure_retrieval_experiments.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{args.output_dir}/figure_retrieval_experiments.pdf", bbox_inches='tight')
        print(f"\nSaved publication figure to {args.output_dir}/")
        
    except Exception as e:
        print(f"Visualization error: {e}")
    
    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
