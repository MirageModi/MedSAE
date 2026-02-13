#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import json
import random
from collections import defaultdict

TOP_N = 10
CONTROL_TERMS = ["hypertension", "patient", "hospital", "treatment"]

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

def get_top_n_set(row):
    try:
        ids = json.loads(row['latent_ids_json'])
        vals = json.loads(row['z_values_json'])
        pairs = sorted(zip(ids, vals), key=lambda x: x[1], reverse=True)
        return set(p[0] for p in pairs[:TOP_N])
    except:
        return set()

def compute_jaccard(set_a, set_b):
    u = len(set_a | set_b)
    return len(set_a & set_b) / u if u > 0 else 0.0

def get_dist(list_a, list_b, n=1000):
    scores = []
    limit = max(10, min(n, len(list_a) * len(list_b)))
    for _ in range(limit):
        a = random.choice(list_a)
        b = random.choice(list_b)
        scores.append(compute_jaccard(a, b))
    return scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk_csv", required=True)
    parser.add_argument("--ner_jsonl", required=True, help="ner_spans.jsonl with char offsets")
    parser.add_argument("--lexicon_csv", required=True)
    parser.add_argument("--output_csv", default="final_jaccard_comparison.csv")
    args = parser.parse_args()

    print("Loading Data...")
    df = pd.read_csv(args.topk_csv, low_memory=False)
    df['core_set'] = df.apply(get_top_n_set, axis=1)
    df['lemma_lower'] = df['lemma'].str.lower()
    df['doc_id'] = df['doc_id'].astype(str)
    
    df = df.dropna(subset=['char_start', 'char_end'])
    df['char_start'] = df['char_start'].astype(int)
    df['char_end'] = df['char_end'].astype(int)
    print(f"Rows with valid char offsets: {len(df)}")

    print("Loading NER spans by character offset...")
    ner_exact = {}
    ner_by_doc = defaultdict(list)
    
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
        
        if (doc_id, char_start) in ner_exact:
            return ner_exact[(doc_id, char_start)]
        
        for span in ner_by_doc.get(doc_id, []):
            if span['end'] > char_start and span['start'] < char_end:
                return span['grp']
        return None

    df['ner_label'] = df.apply(get_ner_label, axis=1)
    print(f"Rows with NER labels: {df['ner_label'].notna().sum()}")

    results = []

    print("\n--- Monosemantic Baseline (Top-10) ---")
    for term in CONTROL_TERMS:
        term_df = df[df['lemma_lower'] == term]
        term_df_with_ner = term_df[term_df['ner_label'].notna()]
        sets = term_df_with_ner['core_set'].tolist()
        if len(sets) < 10: 
            print(f"{term}: skipped (only {len(sets)} samples with NER labels)")
            continue
        
        mid = len(sets)//2
        scores = get_dist(sets[:mid], sets[mid:])
        mean_j = np.mean(scores)
        print(f"{term}: {mean_j:.1%} (n={len(sets)} with NER labels)")
        
        results.append({"type": "Monosemantic", "label": term, "jaccard": mean_j, "n": len(sets)})

    print("\n--- Polysemous Cross-Sense (Top-10) ---")
    lex = pd.read_csv(args.lexicon_csv)
    lex['term_lower'] = lex['term'].str.lower()
    
    ner_map_norm = {}
    for k, v in SENSE_TO_NER_MAP.items():
        if isinstance(v, list): 
            ner_map_norm[k] = [x.lower().strip() for x in v]
        else: 
            ner_map_norm[k] = [v.lower().strip()]

    for term in lex['term_lower'].unique():
        term_df = df[df['lemma_lower'] == term]
        if len(term_df) < 2: continue
        
        sense_sets = defaultdict(list)
        possible_senses = lex[lex['term_lower'] == term]['sense'].unique()
        
        for _, row in term_df.iterrows():
            if row['ner_label'] is None:
                continue
            lbl = str(row['ner_label']).lower().strip()
            matches = [s for s in possible_senses if lbl in ner_map_norm.get(s, [])]
            
            if len(matches) == 1:
                sense_sets[matches[0]].append(row['core_set'])
        
        senses = list(sense_sets.keys())
        for i in range(len(senses)):
            for j in range(i+1, len(senses)):
                s1, s2 = senses[i], senses[j]
                if len(sense_sets[s1]) < 5 or len(sense_sets[s2]) < 5:
                    continue
                scores = get_dist(sense_sets[s1], sense_sets[s2])
                mean_j = np.mean(scores)
                n1, n2 = len(sense_sets[s1]), len(sense_sets[s2])
                print(f"{term} ({s1} vs {s2}): {mean_j:.1%} (n={n1},{n2})")
                
                results.append({
                    "type": "Polysemous", 
                    "label": f"{term} ({s1} vs {s2})", 
                    "jaccard": mean_j,
                    "n": f"{n1},{n2}"
                })

    pd.DataFrame(results).to_csv(args.output_csv, index=False)
    print(f"\nSaved to {args.output_csv}")

if __name__ == "__main__":
    main()