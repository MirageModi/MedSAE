#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import json
import random
import os
from collections import defaultdict

CONTROL_TERMS = [
    "hypertension", "patient", "hospital","treatment"
]

def load_topk_latents_csv(csv_path: str) -> pd.DataFrame:
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    if "latent_ids_json" in df.columns:
        df["latent_ids"] = df["latent_ids_json"].apply(lambda x: json.loads(x) if isinstance(x, str) else [])
    return df

def compute_self_jaccard(df, word, n_samples=2000):
    """
    Splits occurrences of 'word' into random pairs and calculates overlap.
    """
    word_df = df[df["lemma"].str.lower() == word].copy()
    
    # Need at least 2 occurrences to compare
    if len(word_df) < 2:
        return []

    latent_sets = [set(x) for x in word_df["latent_ids"].tolist() if x]
    if len(latent_sets) < 2:
        return []

    scores = []
    # Bootstrap sampling
    for _ in range(n_samples):
        a, b = random.sample(latent_sets, 2)
        intersection = len(a & b)
        union = len(a | b)
        if union > 0:
            scores.append(intersection / union)
            
    return scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk_csv", required=True)
    parser.add_argument("--ner_jsonl", required=True, help="ner_spans.jsonl with char offsets")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    
    df = load_topk_latents_csv(args.topk_csv)
    
    # Filter to valid char offsets
    df = df.dropna(subset=['char_start', 'char_end'])
    df['char_start'] = df['char_start'].astype(int)
    df['char_end'] = df['char_end'].astype(int)
    df['doc_id'] = df['doc_id'].astype(str)
    df['lemma_lower'] = df['lemma'].str.lower()
    print(f"Rows with valid char offsets: {len(df)}")
    
    # Load NER by char offset (same method as jaccard_10_final.py)
    print("Loading NER spans by character offset...")
    ner_exact = {}  # (doc_id, char_start) -> entity_group
    ner_by_doc = defaultdict(list)  # doc_id -> list of spans
    
    with open(args.ner_jsonl) as f:
        for line in f:
            obj = json.loads(line)
            doc_id = str(obj.get('doc_id'))
            grp = obj.get('entity_group')
            start = int(obj.get('start', 0))
            end = int(obj.get('end', 0))
            
            ner_exact[(doc_id, start)] = grp
            ner_by_doc[doc_id].append({'start': start, 'end': end, 'grp': grp})
    
    print(f"Loaded {len(ner_exact)} NER spans")
    
    def get_ner_label(row):
        doc_id = str(row['doc_id'])
        char_start = int(row['char_start'])
        char_end = int(row['char_end'])
        
        # Try exact match first
        if (doc_id, char_start) in ner_exact:
            return ner_exact[(doc_id, char_start)]
        
        # Fall back to overlap search
        for span in ner_by_doc.get(doc_id, []):
            if span['end'] > char_start and span['start'] < char_end:
                return span['grp']
        return None
    
    df['ner_label'] = df.apply(get_ner_label, axis=1)
    print(f"Rows with NER labels: {df['ner_label'].notna().sum()}")
    
    # Filter to only include tokens with NER labels (quality control for consistency)
    df_with_ner = df[df['ner_label'].notna()].copy()
    print(f"Filtered to {len(df_with_ner)} rows with NER labels")
    
    all_scores = []
    summary_rows = []
    
    print(f"{'Term':<15} | {'N (Tokens)':<10} | {'Mean Jaccard':<15}")
    print("-" * 45)
    
    for term in CONTROL_TERMS:
        # Use filtered dataframe with NER labels
        scores = compute_self_jaccard(df_with_ner, term)
        if scores:
            mean_j = np.mean(scores)
            n_tokens = len(df_with_ner[df_with_ner['lemma_lower'] == term])
            print(f"{term:<15} | {n_tokens:<10} | {mean_j:.1%}")
            all_scores.extend(scores)
            summary_rows.append({"term": term, "mean_jaccard": mean_j})
    
    if all_scores:
        baseline_mean = np.mean(all_scores)
        
        # Save raw distribution for ECDF
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, "monosemantic_baseline_dist.csv")
        pd.DataFrame({"jaccard": all_scores}).to_csv(out_path, index=False)
        
        print("-" * 45)
        print(f"Global Monosemantic Baseline: {baseline_mean:.1%}")
        print(f"Saved distribution to {out_path}")
    else:
        print("Error: No control terms found in Top-K CSV. Did you re-run Step 3d with the full lexicon?")

if __name__ == "__main__":
    main()