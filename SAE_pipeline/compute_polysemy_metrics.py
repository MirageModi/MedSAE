#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# Setup logging (will be configured in main())
logger = logging.getLogger(__name__)

# =============================================================================
# Helper Functions
# =============================================================================

def bh_fdr(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction (Legacy SciPy compatible)."""
    pvalues = np.asarray(pvalues)
    n = len(pvalues)
    if n == 0:
        return np.array([])
    
    sorted_indices = np.argsort(pvalues)
    sorted_p = pvalues[sorted_indices]
    
    # Calculate corrected thresholds: q = p * (n / rank)
    ranks = np.arange(1, n + 1)
    qvalues = sorted_p * (n / ranks)
    
    # Enforce monotonicity (q_i <= q_{i+1})
    qvalues = np.minimum.accumulate(qvalues[::-1])[::-1]
    
    # Map back to original order
    original_q = np.zeros_like(qvalues)
    original_q[sorted_indices] = qvalues
    return original_q


SENSE_TO_NER_MAP = {

    # Primary labels first, then fallbacks
    # IMPORTANT: "medication or therapeutic agent" removed from non-therapeutic senses
    # to avoid ambiguous matching
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


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_topk_latents_csv(csv_path: str) -> pd.DataFrame:
    """Load TopK latents CSV and parse JSON columns."""
    df = pd.read_csv(csv_path, low_memory=False)
    if "latent_ids_json" in df.columns:
        df["latent_ids"] = df["latent_ids_json"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else []
        )
    return df


def load_ner_by_char_offset(ner_path: Path) -> Tuple[Dict, Dict]:
    """
    Load NER spans indexed by (doc_id, char_start) for character-based matching.
    
    Returns:
        index_exact: dict mapping (doc_id, char_start) -> span info
        index_by_doc: dict mapping doc_id -> list of spans (for range queries)
    """
    index_exact = {}
    index_by_doc = defaultdict(list)
    
    logger.info(f"Loading NER spans from {ner_path}...")
    
    with ner_path.open() as f:
        for line in tqdm(f, desc="Loading NER"):
            if not line.strip():
                continue
            obj = json.loads(line)
            doc_id = str(obj.get('doc_id', ''))
            start = int(obj.get('start', 0))
            end = int(obj.get('end', 0))
            
            span_info = {
                'entity_group': obj.get('entity_group'),
                'start': start,
                'end': end,
                'score': obj.get('score', 0),
                'text': obj.get('text', ''),
            }
            
            # Index by exact char_start
            key = (doc_id, start)
            if key not in index_exact or span_info['score'] > index_exact[key]['score']:
                index_exact[key] = span_info
            
            # Also store by doc for range queries
            index_by_doc[doc_id].append(span_info)
    
    logger.info(f"  Loaded {len(index_exact):,} unique (doc_id, char_start) keys")
    logger.info(f"  Covering {len(index_by_doc):,} documents")
    
    return index_exact, index_by_doc


def find_ner_for_token(doc_id: str, char_start: int, char_end: int, 
                       index_exact: Dict, index_by_doc: Dict) -> Optional[Dict]:
    """
    Find NER span that covers the given token character range.
    
    Strategy:
    1. Try exact match on (doc_id, char_start)
    2. If no exact match, search for overlapping spans in the document
    """
    # Try exact match first (fast)
    key = (doc_id, char_start)
    if key in index_exact:
        return index_exact[key]
    
    # Fall back to range search (slower but handles partial overlaps)
    if doc_id not in index_by_doc:
        return None
    
    best_span = None
    best_score = -1
    
    for span in index_by_doc[doc_id]:
        # Check for overlap: token [char_start, char_end) overlaps span [start, end)
        if span['end'] <= char_start:
            continue  # Span ends before token starts
        if span['start'] >= char_end:
            continue  # Span starts after token ends
        
        # Overlap exists - keep highest scoring span
        if span['score'] > best_score:
            best_span = span
            best_score = span['score']
    
    return best_span


# =============================================================================
# Metric Computation Functions
# =============================================================================

def compute_entropy_per_sense(
    df: pd.DataFrame, 
    sense_col: str = "sense", 
    latent_ids_col: str = "latent_ids"
) -> pd.DataFrame:
    """Compute entropy of latent distribution per sense."""
    results = []
    group_cols = ["lemma", sense_col]
    if "layer" in df.columns:
        group_cols.append("layer")
    if "model" in df.columns:
        group_cols.append("model")

    for keys, group in df.groupby(group_cols):
        if isinstance(keys, tuple):
            lemma, sense = keys[0], keys[1]
            layer = keys[2] if len(keys) > 2 else None
            model = keys[3] if len(keys) > 3 else None
        else:
            lemma, sense = keys, group[sense_col].iloc[0]
            layer, model = None, None

        all_ids = []
        for lids in group[latent_ids_col]:
            if isinstance(lids, list):
                all_ids.extend(lids)
        
        if not all_ids:
            continue
        
        unique, counts = np.unique(all_ids, return_counts=True)
        probs = counts / len(all_ids)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_ent = np.log2(len(unique) + 1e-10)
        
        results.append({
            "lemma": lemma,
            "sense": sense,
            "entropy": float(entropy),
            "normalized_entropy": float(entropy / max_ent) if max_ent > 0 else 0.0,
            "n_occurrences": len(group),
            "n_unique_latents": len(unique),
            "layer": layer,
            "model": model
        })
    
    return pd.DataFrame(results)


def compute_jaccard_cross_sense(
    df: pd.DataFrame, 
    sense_col: str = "sense", 
    latent_ids_col: str = "latent_ids"
) -> pd.DataFrame:
    """Compute cross-sense Jaccard similarity."""
    results = []
    group_cols = ["lemma"]
    if "layer" in df.columns:
        group_cols.append("layer")
    if "model" in df.columns:
        group_cols.append("model")

    for keys, lemma_df in df.groupby(group_cols):
        if isinstance(keys, tuple):
            lemma = keys[0]
            layer = keys[1] if len(keys) > 1 else None
            model = keys[2] if len(keys) > 2 else None
        else:
            lemma = keys
            layer = lemma_df.get("layer", pd.Series([None])).iloc[0]
            model = lemma_df.get("model", pd.Series([None])).iloc[0]
            
        senses = lemma_df[sense_col].unique()
        if len(senses) < 2:
            continue
        
        lemma_df = lemma_df.reset_index(drop=True)
        
        for i, sense1 in enumerate(senses):
            for sense2 in senses[i+1:]:
                group1 = lemma_df[lemma_df[sense_col] == sense1]
                group2 = lemma_df[lemma_df[sense_col] == sense2]
                
                # Minimum sample requirement: at least 5 samples per sense (same as k=10 analysis)
                if len(group1) < 5 or len(group2) < 5:
                    continue
                
                # Store original sample sizes (for reporting consistency with k=10)
                n_samples_sense1_original = len(group1)
                n_samples_sense2_original = len(group2)
                
                # Sample if groups are large
                g1_idxs = group1.index.tolist()
                g2_idxs = group2.index.tolist()
                
                if len(g1_idxs) > 100:
                    g1_idxs = np.random.choice(g1_idxs, 100, replace=False).tolist()
                if len(g2_idxs) > 100:
                    g2_idxs = np.random.choice(g2_idxs, 100, replace=False).tolist()
                
                # Store sampled sizes (for reporting)
                n_samples_sense1_sampled = len(g1_idxs)
                n_samples_sense2_sampled = len(g2_idxs)
                
                jaccards = []
                for idx1 in g1_idxs:
                    row1 = lemma_df.loc[idx1]
                    set1 = set(row1[latent_ids_col]) if isinstance(row1[latent_ids_col], list) else set()
                    if not set1:
                        continue
                    
                    for idx2 in g2_idxs:
                        row2 = lemma_df.loc[idx2]
                        # Safety check: never compare same document position
                        if row1.get("doc_id") == row2.get("doc_id") and row1.get("position") == row2.get("position"):
                            continue
                            
                        set2 = set(row2[latent_ids_col]) if isinstance(row2[latent_ids_col], list) else set()
                        if not set2:
                            continue
                        
                        union = len(set1 | set2)
                        if union > 0:
                            jaccards.append(len(set1 & set2) / union)
                
                if jaccards:
                    results.append({
                        "lemma": lemma,
                        "sense1": sense1,
                        "sense2": sense2,
                        "mean_jaccard": float(np.mean(jaccards)),
                        "median_jaccard": float(np.median(jaccards)),
                        "std_jaccard": float(np.std(jaccards)),
                        "n_pairs": len(jaccards),
                        "n_samples_sense1": n_samples_sense1_original,
                        "n_samples_sense2": n_samples_sense2_original,
                        "n_samples_sense1_sampled": n_samples_sense1_sampled,
                        "n_samples_sense2_sampled": n_samples_sense2_sampled,
                        "layer": layer,
                        "model": model,
                    })
    
    return pd.DataFrame(results)


def identify_sense_linked_latents(
    df: pd.DataFrame,
    sense_col: str = "sense",
    latent_ids_col: str = "latent_ids",
    fdr_threshold: float = 0.05,
    probe_stats: Optional[pd.DataFrame] = None,
    probe_auroc_threshold: Optional[float] = 0.6,
    probe_ap_threshold: Optional[float] = None,
    presence_min: float = 0.1,
    presence_diff_min: float = 0.05,
) -> pd.DataFrame:
    """
    Identify sense-linked latents using co-occurrence stats and optional probe metrics.
    """
    
    # 1. Filter by Probe AUROC and/or AP if available
    valid_latents = None
    if probe_stats is not None and not probe_stats.empty:
        probe_filtered = probe_stats.copy()
        if probe_auroc_threshold:
            probe_filtered = probe_filtered[probe_filtered["auroc"] >= probe_auroc_threshold]
        if probe_ap_threshold is not None and "ap" in probe_filtered.columns:
            probe_filtered = probe_filtered[probe_filtered["ap"] >= probe_ap_threshold]
        valid_latents = set(probe_filtered["latent_id"].astype(int))
        
        filter_parts = []
        if probe_auroc_threshold:
            filter_parts.append(f"AUROC >= {probe_auroc_threshold}")
        if probe_ap_threshold is not None and "ap" in probe_stats.columns:
            filter_parts.append(f"AP >= {probe_ap_threshold}")
        filter_msg = " and ".join(filter_parts) if filter_parts else "no threshold"
        logger.info(f"  Filtering to {len(valid_latents)} latents with {filter_msg}")

    # 2. Explode Data (Token -> Latent mappings)
    df_exploded = df.explode(latent_ids_col)
    df_exploded = df_exploded.dropna(subset=[latent_ids_col])
    if df_exploded.empty:
        return pd.DataFrame()
    
    df_exploded["latent_id"] = df_exploded[latent_ids_col].astype(int)
    
    # Pre-filter to valid latents if available
    if valid_latents is not None:
        df_exploded = df_exploded[df_exploded["latent_id"].isin(valid_latents)]
        if df_exploded.empty:
            logger.warning("No latents passed probe filtering!")
            return pd.DataFrame()

    # 3. Compute Counts
    # A. Total tokens per (Lemma, Sense)
    sense_totals = df.groupby(["lemma", sense_col]).size().rename("n_sense_total")
    
    # B. Latent hits per (Lemma, Sense, Latent)
    latent_hits = df_exploded.groupby(["lemma", sense_col, "latent_id"]).size().rename("n_hits")
    
    # Join
    stats_df = latent_hits.reset_index().join(sense_totals, on=["lemma", sense_col])
    
    # 4. Calculate P(Latent | Sense)
    stats_df["presence_rate"] = stats_df["n_hits"] / stats_df["n_sense_total"]
    
    # 5. Calculate Specificity (Diff against other senses)
    lemma_latent_hits = df_exploded.groupby(["lemma", "latent_id"]).size().rename("n_lemma_hits")
    lemma_totals = df.groupby("lemma").size().rename("n_lemma_total")
    
    stats_df = stats_df.join(lemma_latent_hits, on=["lemma", "latent_id"])
    stats_df = stats_df.join(lemma_totals, on="lemma")
    
    # "Other" stats
    stats_df["n_other_hits"] = stats_df["n_lemma_hits"] - stats_df["n_hits"]
    stats_df["n_other_total"] = stats_df["n_lemma_total"] - stats_df["n_sense_total"]
    
    # P(Latent | Not Sense)
    stats_df["presence_rate_other"] = np.where(
        stats_df["n_other_total"] > 0,
        stats_df["n_other_hits"] / stats_df["n_other_total"],
        0.0
    )
    
    stats_df["presence_rate_diff"] = stats_df["presence_rate"] - stats_df["presence_rate_other"]
    
    # 6. Apply Thresholds
    filtered = stats_df[
        (stats_df["presence_rate"] >= presence_min) & 
        (stats_df["presence_rate_diff"] >= presence_diff_min)
    ].copy()
    
    if filtered.empty:
        return pd.DataFrame()

    # 7. Fisher Exact Test (Significance)
    pvalues = []
    for _, row in filtered.iterrows():
        table = [
            [int(row["n_hits"]), int(row["n_sense_total"] - row["n_hits"])],
            [int(row["n_other_hits"]), int(row["n_other_total"] - row["n_other_hits"])]
        ]
        if row["n_other_total"] == 0:
            pvalues.append(1.0)
        else:
            _, p = stats.fisher_exact(table, alternative='greater')
            pvalues.append(p)
            
    filtered["pvalue"] = pvalues
    
    # FDR Correction (Benjamini-Hochberg)
    filtered["qvalue"] = 1.0
    for lemma, group in filtered.groupby("lemma"):
        if len(group) > 1:
            qvals = bh_fdr(group["pvalue"].values)
            filtered.loc[group.index, "qvalue"] = qvals
        else:
            filtered.loc[group.index, "qvalue"] = group["pvalue"]
            
    # Filter by FDR threshold
    filtered = filtered[filtered["qvalue"] < fdr_threshold]
    
    # 8. Merge Probe Metadata (if available)
    if probe_stats is not None and not filtered.empty:
        filtered = filtered.merge(probe_stats, on="latent_id", how="left")
        
    return filtered


def sanity_check_latent_signal(csv_path: str, word: str, top_n: int = 10):
    """
    Sanity check to see if any latent is differentiating the senses for a given word.
    """
    logger.info(f"Running sanity check for word '{word}'...")
    
    df = load_topk_latents_csv(csv_path)
    df_word = df[df["lemma"].str.lower() == word.lower()].copy()
    
    if df_word.empty:
        logger.warning(f"No occurrences found for word '{word}'")
        return
    
    exploded = df_word.explode("latent_ids")
    exploded = exploded.dropna(subset=["latent_ids"])
    
    if exploded.empty:
        logger.warning(f"No latent IDs found for word '{word}'")
        return
    
    total_occurrences = len(df_word)
    logger.info(f"Total occurrences of '{word}': {total_occurrences}")
    
    latent_counts = exploded["latent_ids"].value_counts()
    logger.info(f"Top {top_n} most common latents for this word:")
    for latent_id, count in latent_counts.head(top_n).items():
        percentage = (count / total_occurrences) * 100
        logger.info(f"  Latent {latent_id}: {count} occurrences ({percentage:.1f}% of total)")
    
    if "sense" in df_word.columns:
        logger.info("\nBreakdown by sense:")
        for sense, sense_group in df_word.groupby("sense"):
            sense_exploded = sense_group.explode("latent_ids").dropna(subset=["latent_ids"])
            if not sense_exploded.empty:
                top_latent = sense_exploded["latent_ids"].value_counts().head(1)
                if len(top_latent) > 0:
                    latent_id, count = top_latent.index[0], top_latent.iloc[0]
                    logger.info(f"  Sense '{sense}': {len(sense_group)} occurrences, top latent {latent_id} appears {count} times")


# =============================================================================
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute polysemy metrics using character-based NER alignment"
    )
    parser.add_argument("--topk_csv", required=True,
                        help="Path to topk_latents.csv (must have char_start, char_end columns)")
    parser.add_argument("--lexicon_csv", required=True,
                        help="Path to polysemy_lexicon.csv")
    parser.add_argument("--ner_jsonl", required=True,
                        help="Path to ner_spans.jsonl (raw NER with char offsets)")
    parser.add_argument("--output_dir", default="polysemy_metrics",
                        help="Output directory for metrics")
    parser.add_argument("--fdr_threshold", type=float, default=0.05,
                        help="FDR threshold for sense-linked latent identification")
    
    # Probe arguments
    parser.add_argument("--probe_csv", default="",
                        help="Path to probe results CSV (optional)")
    parser.add_argument("--probe_auroc_threshold", type=float, default=0.6,
                        help="Minimum AUROC for probe filtering")
    parser.add_argument("--probe_ap_threshold", type=float, default=0.0,
                        help="Minimum AP for probe filtering (0 = disabled)")
    parser.add_argument("--probe_presence_min", type=float, default=0.1,
                        help="Minimum presence rate for sense-linked latents")
    parser.add_argument("--probe_presence_diff_min", type=float, default=0.05,
                        help="Minimum presence rate difference for sense-linked latents")
    
    # Utility arguments
    parser.add_argument("--log_level", default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Set logging level")
    parser.add_argument("--sanity_check_word", default="",
                        help="Run sanity check for a specific word and exit")
    
    args = parser.parse_args()

    # Configure logging
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s: %(message)s'
    )
    logger.setLevel(log_level)

    # Run sanity check if requested
    if args.sanity_check_word:
        sanity_check_latent_signal(args.topk_csv, args.sanity_check_word)
        return

    # ==========================================================================
    # 1. Load TopK latents
    # ==========================================================================
    logger.info(f"Loading TopK latents from {args.topk_csv}...")
    df_topk = load_topk_latents_csv(args.topk_csv)
    logger.info(f"  Rows: {len(df_topk):,}")
    logger.info(f"  Columns: {list(df_topk.columns)}")
    
    # Check for required char offset columns
    if 'char_start' not in df_topk.columns or 'char_end' not in df_topk.columns:
        logger.error("TopK CSV must have 'char_start' and 'char_end' columns!")
        logger.error("Run fix_topk_char_offsets_v2.py first, or use updated sae_viewer_topk_sharded.py")
        return
    
    # Filter to valid char offsets
    df_topk = df_topk.dropna(subset=['char_start', 'char_end'])
    df_topk['char_start'] = df_topk['char_start'].astype(int)
    df_topk['char_end'] = df_topk['char_end'].astype(int)
    df_topk['doc_id'] = df_topk['doc_id'].astype(str)
    
    # Filter out zero offsets (invalid)
    valid_offsets = (df_topk['char_start'] > 0) | (df_topk['char_end'] > 0)
    df_topk = df_topk[valid_offsets]
    logger.info(f"  Rows with valid char offsets: {len(df_topk):,}")
    
    if df_topk.empty:
        logger.error("No rows with valid char offsets! Check your topk_csv.")
        return

    # ==========================================================================
    # 2. Load NER indexed by character offset
    # ==========================================================================
    ner_path = Path(args.ner_jsonl)
    if not ner_path.exists():
        logger.error(f"NER file not found: {ner_path}")
        return
    
    index_exact, index_by_doc = load_ner_by_char_offset(ner_path)

    # ==========================================================================
    # 3. Load Lexicon and Merge
    # ==========================================================================
    logger.info(f"Loading Lexicon from {args.lexicon_csv}...")
    df_lex = pd.read_csv(args.lexicon_csv)
    df_topk["lemma_lower"] = df_topk["lemma"].str.lower()
    df_lex["term_lower"] = df_lex["term"].str.lower()
    
    # Merge with lexicon (creates duplicates: 1 token -> N senses)
    df_merged = df_topk.merge(df_lex, left_on="lemma_lower", right_on="term_lower", how="inner")
    logger.info(f"After lexicon merge: {len(df_merged):,} rows")

    # ==========================================================================
    # 4. Disambiguate using CHARACTER OFFSET matching with STRICT filtering
    #    (Same as jaccard_10_final.py: len(matches) == 1)
    # ==========================================================================
    logger.info("Disambiguating senses using STRICT character offset matching (len(matches) == 1)...")
    
    disambiguation_stats = defaultdict(int)
    valid_rows = []
    
    def norm(s):
        return str(s).lower().strip() if s else ''
    
    # Build NER map normalized (like jaccard_10_final.py)
    ner_map_norm = {}
    for k, v in SENSE_TO_NER_MAP.items():
        if isinstance(v, list): 
            ner_map_norm[k] = [x.lower().strip() for x in v]
        else: 
            ner_map_norm[k] = [v.lower().strip()]
    
    # Group by term to get possible senses (like k=10 analysis)
    for term in df_lex['term_lower'].unique():
        term_df = df_merged[df_merged['lemma_lower'] == term]
        if len(term_df) < 2:
            continue
        
        possible_senses = df_lex[df_lex['term_lower'] == term]['sense'].unique()
        
        for idx, row in tqdm(term_df.iterrows(), total=len(term_df), desc=f"Processing {term}"):
            doc_id = str(row['doc_id'])
            char_start = int(row['char_start'])
            char_end = int(row['char_end'])
            
            # Find NER span for this token
            ner_span = find_ner_for_token(doc_id, char_start, char_end, index_exact, index_by_doc)
            
            if ner_span is None:
                disambiguation_stats['missing_ner'] += 1
                continue
            
            ner_label = norm(ner_span['entity_group'])
            
            # STRICT MATCHING: Check which senses this NER label could match (like k=10)
            matches = []
            for sense in possible_senses:
                sense_norm = norm(sense)
                expected_labels = ner_map_norm.get(sense_norm, [sense_norm])
                if ner_label in expected_labels:
                    matches.append(sense_norm)
            
            if len(matches) == 1:
                matched_sense = matches[0]
                if matched_sense == norm(row['sense']):
                    disambiguation_stats['match'] += 1
                    row_dict = row.to_dict()
                    row_dict['ner_entity_group'] = ner_span['entity_group']
                    valid_rows.append(row_dict)
                else:
                    disambiguation_stats['sense_mismatch'] += 1
            elif len(matches) == 0:
                disambiguation_stats['no_match'] += 1
            else:
                disambiguation_stats['ambiguous'] += 1 
    
    logger.info(f"\nDisambiguation Stats: {dict(disambiguation_stats)}")
    
    if not valid_rows:
        logger.error("No matches found! Check NER alignment and SENSE_TO_NER_MAP.")
        return
    
    df_clean = pd.DataFrame(valid_rows)
    logger.info(f"Matched rows: {len(df_clean):,}")

    # ==========================================================================
    # 5. Check for Specialty Polysemy Conflicts
    # ==========================================================================
    terms_to_drop = set()
    for term, group in df_clean.groupby("lemma_lower"):
        senses = group["sense"].unique()
        if len(senses) < 2:
            continue
        
        actual_labels = group["ner_entity_group"].apply(norm).unique()
        if len(actual_labels) < 2:
            logger.warning(f"Dropping term '{term}': All senses mapped to same label '{actual_labels}'")
            terms_to_drop.add(term)

    df_clean = df_clean[~df_clean["lemma_lower"].isin(terms_to_drop)]
    logger.info(f"After dropping conflicting terms: {len(df_clean):,} rows")

    # ==========================================================================
    # 6. Ensure latent_ids is parsed
    # ==========================================================================
    if "latent_ids" not in df_clean.columns and "latent_ids_json" in df_clean.columns:
        df_clean["latent_ids"] = df_clean["latent_ids_json"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else []
        )

    # ==========================================================================
    # 7. Create output directory
    # ==========================================================================
    os.makedirs(args.output_dir, exist_ok=True)

    # ==========================================================================
    # 8. Compute Jaccard
    # ==========================================================================
    logger.info("Computing Jaccard similarity...")
    df_jac = compute_jaccard_cross_sense(df_clean)
    jac_path = os.path.join(args.output_dir, "jaccard_cross_sense.csv")
    df_jac.to_csv(jac_path, index=False)
    logger.info(f"  Saved {len(df_jac)} Jaccard pairs to {jac_path}")

    # ==========================================================================
    # 9. Compute Entropy
    # ==========================================================================
    logger.info("Computing entropy per sense...")
    df_ent = compute_entropy_per_sense(df_clean)
    ent_path = os.path.join(args.output_dir, "entropy_per_sense.csv")
    df_ent.to_csv(ent_path, index=False)
    logger.info(f"  Saved {len(df_ent)} entropy records to {ent_path}")

    # ==========================================================================
    # 10. Identify Sense-Linked Latents
    # ==========================================================================
    logger.info("Identifying sense-linked latents...")
    
    # Load Probe Stats if available
    probe_df = None
    if args.probe_csv and os.path.exists(args.probe_csv):
        logger.info(f"  Loading probe stats from {args.probe_csv}...")
        probe_df = pd.read_csv(args.probe_csv)
        logger.info(f"  Loaded {len(probe_df)} probe results")
    
    df_linked = identify_sense_linked_latents(
        df_clean, 
        fdr_threshold=args.fdr_threshold,
        probe_stats=probe_df,
        probe_auroc_threshold=args.probe_auroc_threshold,
        probe_ap_threshold=args.probe_ap_threshold if args.probe_ap_threshold > 0.0 else None,
        presence_min=args.probe_presence_min,
        presence_diff_min=args.probe_presence_diff_min
    )
    
    linked_path = os.path.join(args.output_dir, "sense_linked_latents.csv")
    df_linked.to_csv(linked_path, index=False)
    logger.info(f"  Saved {len(df_linked)} sense-linked latents to {linked_path}")

    # ==========================================================================
    # 11. Summary
    # ==========================================================================
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    
    total_attempts = sum(disambiguation_stats.values())
    match_rate = 100 * disambiguation_stats['match'] / total_attempts if total_attempts > 0 else 0
    
    logger.info(f"Disambiguation:")
    logger.info(f"  - Matches: {disambiguation_stats['match']:,}")
    logger.info(f"  - Missing NER: {disambiguation_stats['missing_ner']:,}")
    logger.info(f"  - No match: {disambiguation_stats.get('no_match', 0):,}")
    logger.info(f"  - Ambiguous (rejected): {disambiguation_stats.get('ambiguous', 0):,}")
    logger.info(f"  - Sense mismatch: {disambiguation_stats.get('sense_mismatch', 0):,}")
    logger.info(f"  - Match rate: {match_rate:.1f}%")
    
    logger.info(f"\nFinal dataset: {len(df_clean):,} tokens")
    
    logger.info(f"\nPer-word breakdown:")
    for lemma in sorted(df_clean['lemma_lower'].unique()):
        sub = df_clean[df_clean['lemma_lower'] == lemma]
        sense_counts = sub['sense'].value_counts().to_dict()
        logger.info(f"  {lemma}: {sense_counts}")
    
    logger.info(f"\nOutputs saved to: {args.output_dir}/")
    logger.info("  - entropy_per_sense.csv")
    logger.info("  - jaccard_cross_sense.csv")
    logger.info("  - sense_linked_latents.csv")
    
    logger.info("\nDone.")


if __name__ == "__main__":
    main()