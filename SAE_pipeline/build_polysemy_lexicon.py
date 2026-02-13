#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_polysemy_lexicon.py — Build polysemy lexicon with sense/context rules.

Creates a CSV/JSON lexicon mapping medical terms to their senses (e.g., cardiac vs GI)
with domain-specific regex rules for context detection.
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd


SEED_LEXICON = [
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
    
    Currently returns the base lexicon unchanged. Future implementation will
    analyze co-occurrence patterns to refine sense definitions.
    """
    return base_lexicon.copy()


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
        
        if exclude_regex and re.search(exclude_regex, context_text, flags):
            continue
        
        if domain_regex and re.search(domain_regex, context_text, flags):
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
        context = f"{section} {text}".strip()
        
        for term in terms:
            pattern = rf"\b{re.escape(term)}\b"
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            
            for match in matches:
                start, end = match.span()
                ctx_start = max(0, start - 100)
                ctx_end = min(len(text), end + 100)
                context_snippet = text[ctx_start:ctx_end]
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
    
    if args.seed_only:
        lexicon = SEED_LEXICON
    else:
        lexicon = expand_lexicon_from_ner(
            SEED_LEXICON,
            args.expand_from_ner if args.expand_from_ner else "",
            args.aligned_spans if args.aligned_spans else ""
        )
    
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
        df = pd.DataFrame(lexicon)
        df.to_csv(f"{output_path}.csv", index=False)
        output_path = Path(f"{output_path}.csv")
    
    print(f"Saved lexicon with {len(lexicon)} entries to {output_path}")
    
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

