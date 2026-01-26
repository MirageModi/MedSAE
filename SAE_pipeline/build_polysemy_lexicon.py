#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_polysemy_lexicon.py — Build polysemy lexicon with sense/context rules.

Creates a CSV/JSON lexicon mapping medical terms to their senses (e.g., cardiac vs GI)
with domain-specific regex rules for context detection.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd


# Default seed lexicon for common medical polysemous terms
# Note: Matching is done on individual token lemmas (lowercased, stripped)
# MedGemma uses SentencePiece/BPE tokenization, so terms may be split into subwords
# The matching logic handles partial matches, so include full terms and common subword pieces
SEED_LEXICON = [
    # Regurgitation: cardiac vs GI
    {
        "term": "regurgitation",
        "sense": "cardiac",
        "domain_regexes": r"(mitral|aortic|valv(e|ular)|murmur|LV|systolic|atrial|ventricular|tricuspid)",
        "exclude_regexes": r"(esophag(eal|us)|GERD|reflux|gastric)",
    },
    {
        "term": "regurgitation",
        "sense": "GI",
        "domain_regexes": r"(esophag(eal|us)|GERD|reflux|gastric|hematemesis)",
        "exclude_regexes": r"(mitral|aortic|valv|tricuspid)",
    },
    # Shock: cardiovascular vs neurological
    {
        "term": "shock",
        "sense": "cardiovascular",
        "domain_regexes": r"(cardiogenic|septic|hypovolemic|anaphylactic|vasodilatory|distributive)",
        "exclude_regexes": r"(electrical|spinal|neurogenic)",
    },
    {
        "term": "shock",
        "sense": "neurological",
        "domain_regexes": r"(spinal|neurogenic)",
        "exclude_regexes": r"(cardiogenic|septic|hypovolemic)",
    },
    # Block: cardiac vs neurological
    {
        "term": "block",
        "sense": "cardiac",
        "domain_regexes": r"(AV|atrioventricular|bundle|branch|heart|conduction|SA|sinoatrial)",
        "exclude_regexes": r"(nerve|anesthesia|epidural|spinal)",
    },
    {
        "term": "block",
        "sense": "neurological",
        "domain_regexes": r"(nerve|spinal|epidural|anesthesia|sympathetic)",
        "exclude_regexes": r"(AV|atrioventricular|bundle|branch|conduction)",
    },
    # Mass: radiological vs anatomical
    {
        "term": "mass",
        "sense": "radiological",
        "domain_regexes": r"(CT|MRI|imaging|tumor|lesion|nodule|scan|radiology)",
        "exclude_regexes": r"(muscle|muscular|body|tissue|lean)",
    },
    {
        "term": "mass",
        "sense": "anatomical",
        "domain_regexes": r"(muscle|muscular|body|tissue|lean|soft)",
        "exclude_regexes": r"(CT|MRI|imaging|tumor|lesion|scan)",
    },
    # Discharge: patient discharge vs bodily discharge
    {
        "term": "discharge",
        "sense": "administrative",
        "domain_regexes": r"(home|hospital|facility|plan|date|disposition|admission)",
        "exclude_regexes": r"(wound|drainage|purulent|vaginal|penile|urethral|nipple)",
    },
    {
        "term": "discharge",
        "sense": "clinical",
        "domain_regexes": r"(wound|drainage|purulent|vaginal|penile|urethral|nipple|exudate)",
        "exclude_regexes": r"(home|hospital|facility|plan|date|disposition)",
    },
    # Arrest: cardiac arrest vs respiratory arrest
    {
        "term": "arrest",
        "sense": "cardiac",
        "domain_regexes": r"(cardiac|heart|CPR|defibrillat|asystole|VF|ventricular)",
        "exclude_regexes": r"(respiratory|breathing|ventilation|apnea)",
    },
    {
        "term": "arrest",
        "sense": "respiratory",
        "domain_regexes": r"(respiratory|breathing|ventilation|apnea|respiratory)",
        "exclude_regexes": r"(cardiac|heart|CPR|defibrillat)",
    },
    # Attack: heart attack vs asthma attack
    {
        "term": "attack",
        "sense": "cardiac",
        "domain_regexes": r"(heart|myocardial|MI|infarct|coronary|chest pain)",
        "exclude_regexes": r"(asthma|bronchospasm|wheezing|respiratory)",
    },
    {
        "term": "attack",
        "sense": "respiratory",
        "domain_regexes": r"(asthma|bronchospasm|wheezing|respiratory|dyspnea)",
        "exclude_regexes": r"(heart|myocardial|MI|infarct|coronary)",
    },
    # Episode: seizure episode vs cardiac episode
    {
        "term": "episode",
        "sense": "neurological",
        "domain_regexes": r"(seizure|convulsion|epileptic|syncope|neurological)",
        "exclude_regexes": r"(cardiac|arrhythmia|chest pain|MI)",
    },
    {
        "term": "episode",
        "sense": "cardiac",
        "domain_regexes": r"(cardiac|arrhythmia|chest pain|MI|palpitation)",
        "exclude_regexes": r"(seizure|convulsion|epileptic|syncope)",
    },
    # Syndrome: various medical syndromes
    {
        "term": "syndrome",
        "sense": "cardiac",
        "domain_regexes": r"(coronary|acute coronary|ACS|cardiac|myocardial)",
        "exclude_regexes": r"(respiratory|metabolic|renal|hepatic)",
    },
    {
        "term": "syndrome",
        "sense": "respiratory",
        "domain_regexes": r"(respiratory|distress|ARDS|acute respiratory|breathing)",
        "exclude_regexes": r"(cardiac|coronary|metabolic|renal)",
    },
    # Lesion: skin lesion vs brain lesion
    {
        "term": "lesion",
        "sense": "dermatological",
        "domain_regexes": r"(skin|cutaneous|dermat|rash|ulcer|wound)",
        "exclude_regexes": r"(brain|cerebral|CNS|neurological|MRI brain)",
    },
    {
        "term": "lesion",
        "sense": "neurological",
        "domain_regexes": r"(brain|cerebral|CNS|neurological|MRI brain|CT head)",
        "exclude_regexes": r"(skin|cutaneous|dermat|rash)",
    },
    # Effusion: pleural effusion vs joint effusion
    {
        "term": "effusion",
        "sense": "respiratory",
        "domain_regexes": r"(pleural|lung|chest|thoracic|pneumothorax)",
        "exclude_regexes": r"(joint|knee|shoulder|synovial|arthrocentesis)",
    },
    {
        "term": "effusion",
        "sense": "orthopedic",
        "domain_regexes": r"(joint|knee|shoulder|synovial|arthrocentesis|arthritic)",
        "exclude_regexes": r"(pleural|lung|chest|thoracic)",
    },
    # Infiltrate: pulmonary infiltrate vs cellular infiltrate
    {
        "term": "infiltrate",
        "sense": "respiratory",
        "domain_regexes": r"(pulmonary|lung|chest|X-ray|CXR|pneumonia)",
        "exclude_regexes": r"(cellular|tissue|biopsy|pathology)",
    },
    {
        "term": "infiltrate",
        "sense": "pathological",
        "domain_regexes": r"(cellular|tissue|biopsy|pathology|histology)",
        "exclude_regexes": r"(pulmonary|lung|chest|X-ray|CXR)",
    },
    # Pressure: blood pressure vs intracranial pressure
    {
        "term": "pressure",
        "sense": "cardiovascular",
        "domain_regexes": r"(blood|BP|hypertension|hypotension|systolic|diastolic)",
        "exclude_regexes": r"(intracranial|ICP|brain|CSF|cranial)",
    },
    {
        "term": "pressure",
        "sense": "neurological",
        "domain_regexes": r"(intracranial|ICP|brain|CSF|cranial|ventricular)",
        "exclude_regexes": r"(blood|BP|hypertension|hypotension)",
    },
    # Failure: heart failure vs renal failure
    {
        "term": "failure",
        "sense": "cardiac",
        "domain_regexes": r"(heart|cardiac|CHF|congestive|ejection fraction|LVEF)",
        "exclude_regexes": r"(renal|kidney|AKI|dialysis|creatinine)",
    },
    {
        "term": "failure",
        "sense": "renal",
        "domain_regexes": r"(renal|kidney|AKI|dialysis|creatinine|BUN|nephro)",
        "exclude_regexes": r"(heart|cardiac|CHF|congestive)",
    },
    # Insufficiency: renal insufficiency vs adrenal insufficiency
    {
        "term": "insufficiency",
        "sense": "renal",
        "domain_regexes": r"(renal|kidney|creatinine|BUN|nephro|GFR)",
        "exclude_regexes": r"(adrenal|cortisol|Addison|steroid)",
    },
    {
        "term": "insufficiency",
        "sense": "endocrine",
        "domain_regexes": r"(adrenal|cortisol|Addison|steroid|endocrine)",
        "exclude_regexes": r"(renal|kidney|creatinine|BUN)",
    },
]


def expand_lexicon_from_ner(
    base_lexicon: List[Dict],
    ner_spans_path: str,
    aligned_spans_path: Optional[str] = None,
    min_occurrences: int = 5
) -> List[Dict]:
    """
    Expand lexicon using co-occurring disease entities from NER.
    
    For each term in base_lexicon, find contexts where it co-occurs with
    disease entities and use those to refine sense definitions.
    """
    expanded = base_lexicon.copy()
    
    try:
        if aligned_spans_path and Path(aligned_spans_path).exists():
            # Use aligned spans if available
            df_aligned = pd.read_csv(aligned_spans_path)
            # Group by term and sense, count co-occurring entities
            # (implementation depends on your NER output format)
            pass
        elif ner_spans_path and Path(ner_spans_path).exists():
            # Use raw NER spans
            if ner_spans_path.endswith(".csv"):
                df_ner = pd.read_csv(ner_spans_path)
            elif ner_spans_path.endswith(".json") or ner_spans_path.endswith(".jsonl"):
                with open(ner_spans_path) as f:
                    if ner_spans_path.endswith(".jsonl"):
                        ner_data = [json.loads(line) for line in f if line.strip()]
                    else:
                        ner_data = json.load(f)
                df_ner = pd.DataFrame(ner_data)
            else:
                df_ner = None
            
            if df_ner is not None and not df_ner.empty:
                # Extract co-occurrence patterns
                # (This is a placeholder; actual implementation would analyze
                # document-level co-occurrence of terms with disease entities)
                pass
    except Exception as e:
        print(f"Warning: Could not expand lexicon from NER: {e}")
    
    return expanded


def detect_sense_from_context(
    term: str,
    context_text: str,
    lexicon: List[Dict],
    case_sensitive: bool = False
) -> Optional[str]:
    """
    Detect sense of a term in context using regex rules.
    
    Returns:
        Sense label if detected, None otherwise
    """
    flags = 0 if case_sensitive else re.IGNORECASE
    
    for entry in lexicon:
        if entry["term"].lower() != term.lower():
            continue
        
        domain_regex = entry.get("domain_regexes", "")
        exclude_regex = entry.get("exclude_regexes", "")
        
        # Check exclusion first
        if exclude_regex:
            if re.search(exclude_regex, context_text, flags):
                continue
        
        # Check domain match
        if domain_regex:
            if re.search(domain_regex, context_text, flags):
                return entry["sense"]
    
    return None


def label_occurrences_from_sections(
    lexicon_path: str,
    csv_path: str,
    text_column: str,
    section_column: Optional[str] = None,
    output_path: str = "labeled_occurrences.csv"
):
    """
    Label term occurrences in a CSV using section headers and context rules.
    
    Creates a CSV with columns: doc_id, term, sense, position, context
    """
    df = pd.read_csv(csv_path)
    lexicon = load_lexicon(lexicon_path)
    
    # Get unique terms from lexicon
    terms = list(set(entry["term"] for entry in lexicon))
    
    labeled = []
    for idx, row in df.iterrows():
        text = str(row[text_column])
        section = str(row.get(section_column, "")) if section_column else ""
        
        # Combine section and text for context
        context = f"{section} {text}".strip()
        
        # Search for each term
        for term in terms:
            # Simple word boundary search (can be enhanced with regex)
            pattern = rf"\b{re.escape(term)}\b"
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            
            for match in matches:
                start, end = match.span()
                # Get surrounding context (100 chars before/after)
                ctx_start = max(0, start - 100)
                ctx_end = min(len(text), end + 100)
                context_snippet = text[ctx_start:ctx_end]
                
                # Detect sense
                sense = detect_sense_from_context(term, context_snippet, lexicon)
                
                if sense:
                    labeled.append({
                        "doc_id": row.get("doc_id", idx),
                        "term": term,
                        "sense": sense,
                        "char_start": start,
                        "char_end": end,
                        "context": context_snippet,
                        "section": section,
                    })
    
    df_labeled = pd.DataFrame(labeled)
    df_labeled.to_csv(output_path, index=False)
    print(f"Labeled {len(df_labeled)} occurrences, saved to {output_path}")
    return df_labeled


def load_lexicon(lexicon_path: str) -> List[Dict]:
    """Load lexicon from CSV or JSON."""
    path = Path(lexicon_path)
    if not path.exists():
        print(f"Warning: Lexicon file not found, using seed lexicon")
        return SEED_LEXICON
    
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        return df.to_dict("records")
    elif path.suffix in [".json", ".jsonl"]:
        with open(path) as f:
            if path.suffix == ".jsonl":
                return [json.loads(line) for line in f if line.strip()]
            else:
                return json.load(f)
    else:
        raise ValueError(f"Unsupported lexicon format: {path.suffix}")


def main():
    parser = argparse.ArgumentParser(description="Build polysemy lexicon")
    parser.add_argument("--output", type=str, default="polysemy_lexicon.csv",
                       help="Output lexicon path (CSV/JSON)")
    parser.add_argument("--seed_only", action="store_true",
                       help="Use only seed lexicon, don't expand from NER")
    parser.add_argument("--expand_from_ner", type=str, default="",
                       help="Path to NER spans to expand lexicon")
    parser.add_argument("--aligned_spans", type=str, default="",
                       help="Path to aligned spans (from align_spans_to_tokens.py)")
    parser.add_argument("--label_occurrences", action="store_true",
                       help="Label occurrences in CSV and save to output")
    parser.add_argument("--csv_path", type=str,
                       help="CSV path for labeling occurrences")
    parser.add_argument("--text_column", type=str, default="text",
                       help="Column name with text in CSV")
    parser.add_argument("--section_column", type=str, default="",
                       help="Optional column name with section headers")
    
    args = parser.parse_args()
    
    # Build lexicon
    if args.seed_only:
        lexicon = SEED_LEXICON
    else:
        lexicon = expand_lexicon_from_ner(
            SEED_LEXICON,
            args.expand_from_ner if args.expand_from_ner else "",
            args.aligned_spans if args.aligned_spans else ""
        )
    
    # Save lexicon
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == ".csv":
        df = pd.DataFrame(lexicon)
        df.to_csv(output_path, index=False)
    elif output_path.suffix in [".json", ".jsonl"]:
        with open(output_path, "w") as f:
            if output_path.suffix == "jsonl":
                for entry in lexicon:
                    f.write(json.dumps(entry) + "\n")
            else:
                json.dump(lexicon, f, indent=2)
    else:
        # Default to CSV
        df = pd.DataFrame(lexicon)
        df.to_csv(f"{output_path}.csv", index=False)
        output_path = Path(f"{output_path}.csv")
    
    print(f"Saved lexicon with {len(lexicon)} entries to {output_path}")
    
    # Optionally label occurrences
    if args.label_occurrences and args.csv_path:
        occurrences_path = str(output_path).replace(".csv", "_occurrences.csv").replace(".json", "_occurrences.csv")
        label_occurrences_from_sections(
            str(output_path),
            args.csv_path,
            args.text_column,
            args.section_column if args.section_column else None,
            occurrences_path
        )


if __name__ == "__main__":
    main()

