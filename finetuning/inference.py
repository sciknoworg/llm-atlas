"""
Run ORKG extraction using a LoRA-finetuned model.

Loads the base model + LoRA adapter, processes papers chunk-by-chunk
using the same fixed instruction, and writes one extraction JSON per paper
(same format as the baseline pipeline).

Usage
-----
    # Single paper
    python -m finetuning.inference \\
        --adapter finetuning/output/default/final_adapter \\
        --arxiv-id 2302.13971v1

    # All test-set papers
    python -m finetuning.inference \\
        --adapter finetuning/output/default/final_adapter \\
        --all-test
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from finetuning.config import (
    CHUNK_OVERLAP,
    DATASET_TEST,
    INSTRUCTION,
    MAX_CHUNK_SIZE,
    ORKG_FIELDS,
    PAPERS_DIR,
    PAPERS_LIST_PATH,
    RESULTS_DIR,
    TrainingConfig,
)
from src.pdf_parser import PDFParser

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s"
)
logger = logging.getLogger(__name__)


# ── model loading ────────────────────────────────────────────────────────────


def load_finetuned_model(adapter_path: str, cfg: TrainingConfig):
    """Load base model + LoRA adapter for inference."""

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if cfg.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    logger.info("Loaded fine-tuned model from %s", adapter_path)
    return model, tokenizer


# ── extraction ───────────────────────────────────────────────────────────────


def extract_from_chunk(
    chunk: str, model, tokenizer, max_new_tokens: int = 1024
) -> Optional[Dict[str, Any]]:
    """Run the fine-tuned model on one chunk and parse the JSON output."""

    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": chunk},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return _parse_json(response)


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON extraction from model output."""
    import re

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract JSON block
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()

    # Find outermost braces
    start = text.find("{")
    if start < 0:
        return None
    brace = 0
    end = start
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            brace += 1
        elif ch == "}":
            brace -= 1
            if brace == 0:
                end = i + 1
                break
    if end <= start:
        end = text.rfind("}") + 1

    if end > start:
        candidate = text[start:end]
        # Remove trailing commas
        candidate = re.sub(r",\s*}", "}", candidate)
        candidate = re.sub(r",\s*]", "]", candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


def _merge_chunk_extractions(
    chunk_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge extractions from multiple chunks of the same paper.

    For each field, keep the first non-empty value across chunks.
    """
    merged: Dict[str, Any] = {f: "" for f in ORKG_FIELDS}
    for result in chunk_results:
        for field in ORKG_FIELDS:
            val = result.get(field, "")
            if val and not merged[field]:
                merged[field] = val
    return merged


def extract_paper(
    pdf_path: Path,
    model,
    tokenizer,
    paper_title: str = "",
    arxiv_id: str = "",
) -> Dict[str, Any]:
    """Full extraction pipeline for one paper: parse → chunk → extract → merge."""

    parser = PDFParser(method="pdfplumber", extract_tables=True)
    parsed = parser.parse(pdf_path)
    if parsed is None or not parsed.get("cleaned_text"):
        return {"status": "parse_error", "paper_title": paper_title}

    text = parsed["cleaned_text"]
    chunks = parser.chunk_text(text, max_chunk_size=MAX_CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    chunk_results = []
    for i, chunk in enumerate(chunks):
        logger.info("  Chunk %d/%d (%d chars)", i + 1, len(chunks), len(chunk))
        result = extract_from_chunk(chunk, model, tokenizer)
        if result:
            chunk_results.append(result)

    if not chunk_results:
        return {"status": "no_extraction", "paper_title": paper_title}

    merged = _merge_chunk_extractions(chunk_results)

    return {
        "status": "completed",
        "paper_title": paper_title,
        "arxiv_id": arxiv_id,
        "extraction_data": [merged],
        "chunks_processed": len(chunks),
        "chunks_with_output": len(chunk_results),
        "timestamp": datetime.now().isoformat(),
    }


# ── batch helpers ────────────────────────────────────────────────────────────


def _find_pdf(paper_title: str, arxiv_id: Optional[str]) -> Optional[Path]:
    if arxiv_id:
        clean_id = arxiv_id.replace("/", "_").split("v")[0]
        for pat in (f"{clean_id}*.pdf", f"{arxiv_id.replace('/', '_')}*.pdf"):
            hits = list(PAPERS_DIR.glob(pat))
            if hits:
                return hits[0]
    slug = "".join(c if c.isalnum() else "_" for c in paper_title.lower())[:60]
    for pdf in PAPERS_DIR.glob("*.pdf"):
        if slug[:30] in pdf.stem.lower().replace("-", "_"):
            return pdf
    return None


def _load_test_papers() -> List[Dict[str, str]]:
    """Load paper titles from the test split, then resolve arxiv_ids."""
    if not DATASET_TEST.exists():
        logger.error("Test set not found at %s. Run prepare_dataset first.", DATASET_TEST)
        return []

    test_titles = set()
    with open(DATASET_TEST, "r", encoding="utf-8") as fp:
        for line in fp:
            rec = json.loads(line.strip())
            # The output JSON contains model_name which helps identify the paper,
            # but the input text is the chunk – we need the paper title from the
            # papers_list cross-reference.  For now, collect unique input chunks
            # and match them to papers_list.
            pass

    # Fallback: use papers_list and pick the test fraction
    # Re-derive test papers using the same seed as prepare_dataset
    import random
    from finetuning.config import GOLD_STANDARD_PATH
    from collections import defaultdict

    gold_raw = json.loads(GOLD_STANDARD_PATH.read_text(encoding="utf-8"))
    gold_models = gold_raw.get("extraction_data", gold_raw)
    by_paper = defaultdict(list)
    for m in gold_models:
        by_paper[m.get("paper_title", "unknown")].append(m)

    cfg = TrainingConfig()
    paper_titles = sorted(by_paper.keys())
    random.seed(cfg.seed)
    random.shuffle(paper_titles)
    n_test = max(1, int(len(paper_titles) * cfg.test_ratio))
    test_titles = set(paper_titles[:n_test])

    papers_list = json.loads(PAPERS_LIST_PATH.read_text(encoding="utf-8"))
    title_to_info = {p["paper_title"]: p for p in papers_list}

    result = []
    for title in test_titles:
        info = title_to_info.get(title, {})
        result.append({
            "paper_title": title,
            "arxiv_id": info.get("arxiv_id", ""),
        })
    return result


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Run ORKG extraction with fine-tuned model")
    ap.add_argument("--adapter", type=str, required=True,
                     help="Path to LoRA adapter directory")
    ap.add_argument("--arxiv-id", type=str, default=None,
                     help="Extract a single paper by ArXiv ID")
    ap.add_argument("--all-test", action="store_true",
                     help="Extract all papers from the test split")
    ap.add_argument("--output-dir", type=str, default=None,
                     help="Output directory (default: finetuning/results/<run_name>/)")
    args = ap.parse_args()

    cfg_path = Path(args.adapter).parent / "training_config.json"
    cfg = TrainingConfig()
    if cfg_path.exists():
        import dataclasses
        saved = json.loads(cfg_path.read_text(encoding="utf-8"))
        for f in dataclasses.fields(cfg):
            if f.name in saved:
                setattr(cfg, f.name, saved[f.name])

    model, tokenizer = load_finetuned_model(args.adapter, cfg)

    out_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR / Path(args.adapter).parent.name
    out_dir.mkdir(parents=True, exist_ok=True)

    papers = []
    if args.arxiv_id:
        papers = [{"paper_title": "", "arxiv_id": args.arxiv_id}]
    elif args.all_test:
        papers = _load_test_papers()
    else:
        ap.error("Specify --arxiv-id or --all-test")

    logger.info("Extracting %d paper(s) …", len(papers))

    for i, paper in enumerate(papers, 1):
        title = paper["paper_title"]
        arxiv_id = paper.get("arxiv_id", "")
        logger.info("[%d/%d] %s", i, len(papers), title[:80] or arxiv_id)

        pdf_path = _find_pdf(title, arxiv_id)
        if pdf_path is None:
            logger.warning("  PDF not found, skipping")
            continue

        result = extract_paper(pdf_path, model, tokenizer, title, arxiv_id)

        slug = arxiv_id.replace("/", "_") if arxiv_id else title[:40].replace(" ", "_")
        out_file = out_dir / f"{slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_file, "w", encoding="utf-8") as fp:
            json.dump(result, fp, indent=2, ensure_ascii=False)
        logger.info("  Saved → %s", out_file)

    logger.info("Done. Results in %s", out_dir)


if __name__ == "__main__":
    main()
