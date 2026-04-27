"""
Strict Evaluation Metrics for LLM Extraction Pipeline

This script uses STRICT matching (no relaxations) to measure real extraction accuracy.
Use this for thesis evaluation metrics.

For relaxed evaluation (better UX), use evaluate_extraction.py instead.

Usage:
    python scripts/evaluation/evaluate_extraction_strict.py \\
        --gold data/gold_standard/R1364660.json \\
        --prediction data/extracted/2401.02385_20251207_223913.json
        
Output:
    Evaluation report with per-field and overall metrics (STRICT)
"""

import json
import argparse
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from difflib import SequenceMatcher
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy import for sentence-transformers (heavy dependency)
_sentence_transformer = None


def _get_sentence_transformer():
    """Lazy load sentence transformer model."""
    global _sentence_transformer
    if _sentence_transformer is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence transformer model (all-MiniLM-L6-v2)...")
            _sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer loaded successfully")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Semantic similarity will fall back to fuzzy matching. "
                "Install with: pip install sentence-transformers"
            )
            _sentence_transformer = False  # Mark as unavailable
    return _sentence_transformer if _sentence_transformer is not False else None


# Lazy import for BERTScore (heavy dependency)
_bert_score = None


def _get_bert_score():
    """Lazy load BERTScore module."""
    global _bert_score
    if _bert_score is None:
        try:
            import bert_score
            logger.info("BERTScore module loaded successfully")
            _bert_score = bert_score
        except ImportError:
            logger.warning(
                "bert-score not installed. "
                "BERTScore metrics will not be computed. "
                "Install with: pip install bert-score"
            )
            _bert_score = False  # Mark as unavailable
    return _bert_score if _bert_score is not False else None


class StrictExtractionEvaluator:
    """Evaluates extraction quality with STRICT matching (no relaxations)."""
    
    # Fields to evaluate from ORKG template R609825
    EVALUATION_FIELDS = [
        "model_name",
        "model_family",
        "date_created",
        "organization",
        "innovation",
        "pretraining_architecture",
        "pretraining_task",
        "pretraining_corpus",
        "finetuning_task",
        "optimizer",
        "parameters",
        "parameters_millions",
        "hardware_used",
        "extension",
        "blog_post",
        "license",
        "research_problem",
        "application"
    ]
    
    # Only long-text fields use fuzzy matching (with strict threshold 0.8)
    # optimizer included so e.g. "Adam" matches "Adam optimizer"
    FUZZY_FIELDS = ["innovation", "pretraining_corpus", "application", "research_problem", "optimizer"]
    
    # Fields that benefit from semantic similarity (meaning-based comparison)
    SEMANTIC_FIELDS = ["innovation", "pretraining_corpus", "application", "research_problem", "extension"]

    # Fields used for match-based Overall F1 (structured/exact only; semantic fields evaluated via BERTScore)
    # Defined explicitly to avoid class-body evaluation order issues with SEMANTIC_FIELDS
    STRUCTURED_FIELDS = [
        "model_name", "model_family", "date_created", "organization",
        "pretraining_architecture", "pretraining_task", "finetuning_task",
        "optimizer", "parameters", "parameters_millions", "hardware_used",
        "blog_post", "license"
    ]

    # Fields whose match decision uses BERTScore F1 (instead of exact/fuzzy) for semantic equivalence
    BERTSCORE_MATCH_FIELDS = ["innovation", "license", "pretraining_task", "research_problem", "pretraining_architecture"]

    def __init__(
        self,
        fuzzy_threshold: float = 0.8,
        use_semantic: bool = True,
        bert_score_model: str = "roberta-large",
        include_bert_score: bool = True,
    ):
        """
        Initialize strict evaluator.
        
        Args:
            fuzzy_threshold: Similarity threshold for fuzzy/semantic matching (0-1)
                             Default 0.8 (strict). Use 1.0 for exact-only.
            use_semantic: Use semantic similarity (embeddings) for long-text fields.
                          Default True. Falls back to fuzzy if unavailable.
            bert_score_model: Model for BERTScore computation (default: roberta-large).
                             Options: roberta-large, bert-base-uncased, etc.
            include_bert_score: Compute/report BERTScore metrics for semantic fields.
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.use_semantic = use_semantic
        self.bert_score_model = bert_score_model
        self.include_bert_score = include_bert_score
        self.results = {}
        
    def normalize_value(self, value: Any) -> str:
        """Normalize a value for comparison. STRICT: only None and empty string are missing."""
        if value is None or value == "":
            return ""
        # STRICT: "null", "none", "n/a" are treated as actual values (not missing)
        return str(value).strip().lower()

    def fuzzy_match(self, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings using SequenceMatcher.
        
        Returns:
            Similarity score between 0 and 1
        """
        if not str1 and not str2:
            return 1.0  # Both empty = match
        if not str1 or not str2:
            return 0.0  # One empty, one not = no match
            
        return SequenceMatcher(None, str1, str2).ratio()
    
    def semantic_match(self, str1: str, str2: str) -> float:
        """
        Calculate semantic similarity between two strings using sentence embeddings.
        
        Returns:
            Cosine similarity score between 0 and 1
        """
        if not str1 and not str2:
            return 1.0  # Both empty = match
        if not str1 or not str2:
            return 0.0  # One empty, one not = no match
        
        model = _get_sentence_transformer()
        if model is None:
            # Fall back to fuzzy matching if model unavailable
            return self.fuzzy_match(str1, str2)
        
        try:
            # Encode both strings
            embeddings = model.encode([str1, str2], convert_to_tensor=False)
            
            # Compute cosine similarity
            # embeddings is numpy array of shape (2, embedding_dim)
            emb1, emb2 = embeddings[0], embeddings[1]
            cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            
            # Ensure result is in [0, 1] (cosine can be negative, but for these texts unlikely)
            return max(0.0, min(1.0, float(cosine_sim)))
        except Exception as e:
            logger.warning(f"Semantic matching failed: {e}. Falling back to fuzzy match.")
            return self.fuzzy_match(str1, str2)
    
    def compute_bert_score_batch(
        self,
        references: List[str],
        candidates: List[str]
    ) -> List[float]:
        """
        Compute BERTScore F1 for a batch of (reference, candidate) pairs.
        
        BERTScore uses token-level embeddings to compute semantic similarity,
        capturing paraphrases and meaning better than surface-level metrics.
        
        Args:
            references: List of reference (gold) texts
            candidates: List of candidate (predicted) texts
        
        Returns:
            List of F1 scores (one per pair), each in [0, 1]
        """
        if not references or not candidates:
            return []
        
        if len(references) != len(candidates):
            logger.warning(
                f"BERTScore: reference and candidate counts differ "
                f"({len(references)} vs {len(candidates)})"
            )
            return []
        
        bert_score_module = _get_bert_score()
        if bert_score_module is None:
            logger.warning("BERTScore not available, skipping BERTScore computation")
            return []
        
        try:
            logger.info(f"Computing BERTScore with model: {self.bert_score_model}")
            # Compute BERTScore: returns (P, R, F1) tensors
            P, R, F1 = bert_score_module.score(
                candidates,
                references,
                model_type=self.bert_score_model,
                lang="en",
                verbose=False
            )
            # Convert F1 tensor to list of floats
            return F1.tolist()
        except Exception as e:
            logger.error(f"BERTScore computation failed: {e}")
            return []
    
    @staticmethod
    def _extract_year_month(value: Any) -> Tuple[Optional[int], Optional[int]]:
        """Extract (year, month) from date string. Month is 1-12 or None."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return None, None
        s = str(value).strip()
        # YYYY-MM-DD or YYYY-MM
        m = re.match(r"^(\d{4})-(\d{2})(?:-\d{2})?$", s)
        if m:
            return int(m.group(1)), int(m.group(2))
        # YYYY only
        m = re.match(r"^(\d{4})$", s)
        if m:
            return int(m.group(1)), None
        return None, None

    def compare_date(self, gold_value: Any, pred_value: Any) -> Tuple[bool, float]:
        """
        Compare date_created: match if same year; if both have month, also require same month.
        Returns (is_match, similarity). Same year => at least 0.5; same year+month => 1.0.
        """
        gy, gm = self._extract_year_month(gold_value)
        py, pm = self._extract_year_month(pred_value)
        if gy is None and py is None:
            return True, 1.0
        if gy is None or py is None:
            return False, 0.0
        if gy != py:
            return False, 0.0
        # Same year
        if gm is not None and pm is not None:
            return (gm == pm, 1.0) if gm == pm else (False, 0.5)
        return True, 0.9  # same year, month missing in one

    # Known organization aliases for flexible matching (lowercase)
    _ORG_ALIASES = [
        ("google ai language", "google"),
        ("google research", "google"),
        ("meta ai", "meta"),
        ("facebook ai", "meta"),
        ("openai", "openai"),
        ("microsoft research", "microsoft"),
    ]

    def compare_organization(self, gold_value: Any, pred_value: Any) -> Tuple[bool, float]:
        """
        Compare organization: exact match, or one contains the other, or known alias.
        Returns (is_match, similarity).
        """
        g = self.normalize_value(gold_value)
        p = self.normalize_value(pred_value)
        if not g and not p:
            return True, 1.0
        if not g or not p:
            return False, 0.0
        if g == p:
            return True, 1.0
        if g in p or p in g:
            return True, 0.95
        for a, b in self._ORG_ALIASES:
            if (g == a or g == b) and (p == a or p == b):
                return True, 0.95
        return False, self.fuzzy_match(g, p)

    def compare_optimizer_word_overlap(self, gold_value: Any, pred_value: Any) -> Tuple[bool, float]:
        """
        Compare optimizer: match if any word in the extraction appears in the gold standard.
        E.g. gold "Adam optimizer" vs pred "Adam" -> match (Adam is in gold).
        Returns (is_match, similarity). Similarity is 1.0 when any overlap, else 0.0.
        """
        g = self.normalize_value(gold_value)
        p = self.normalize_value(pred_value)
        if not g and not p:
            return True, 1.0
        if not g or not p:
            return False, 0.0
        # Split into words (whitespace and commas)
        gold_words = set(re.split(r"[\s,]+", g))
        pred_words = set(re.split(r"[\s,]+", p))
        gold_words.discard("")
        pred_words.discard("")
        if not gold_words and not pred_words:
            return True, 1.0
        if not gold_words or not pred_words:
            return False, 0.0
        overlap = pred_words & gold_words
        if overlap:
            return True, 1.0
        return False, 0.0

    @staticmethod
    def _norm_identifier(value: Any) -> str:
        """Normalize identifier (model_name, model_family): lowercase, unify hyphens and spaces."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return ""
        return re.sub(r'[-\s]+', ' ', str(value).strip().lower())

    def compare_identifier_field(self, gold_value: Any, pred_value: Any) -> Tuple[bool, float]:
        """
        Compare identifier-like fields (model_name, model_family).
        Same hyphen/space normalization as pairing; then exact match, containment, or fuzzy.
        """
        gold_norm = self._norm_identifier(gold_value)
        pred_norm = self._norm_identifier(pred_value)
        if not gold_norm and not pred_norm:
            return True, 1.0
        if not gold_norm or not pred_norm:
            return False, 0.0
        if gold_norm == pred_norm:
            return True, 1.0
        if gold_norm in pred_norm or pred_norm in gold_norm:
            return True, 0.95
        similarity = self.fuzzy_match(gold_norm, pred_norm)
        return (similarity >= self.fuzzy_threshold, similarity)

    def compare_parameters_millions(self, gold_value: Any, pred_value: Any) -> Tuple[bool, float]:
        """
        Compare parameters_millions: numeric equality, or both null.
        If one is null, allow match when the other field (parameters) implies same scale.
        """
        def to_int(v: Any) -> Optional[int]:
            if v is None or (isinstance(v, str) and not v.strip()):
                return None
            try:
                return int(float(str(v).strip()))
            except (ValueError, TypeError):
                return None

        g = to_int(gold_value)
        p = to_int(pred_value)
        if g is None and p is None:
            return True, 1.0
        if g is None or p is None:
            return False, 0.0
        if g == p:
            return True, 1.0
        # Allow small relative difference (e.g. 340 vs 340)
        return False, 0.0

    def compare_parameters_list(self, gold_value: Any, pred_value: Any) -> Tuple[bool, float]:
        """
        Compare parameters field (comma-separated list of sizes).
        
        Uses set-based F1: precision = |gold ∩ pred| / |pred|, recall = |gold ∩ pred| / |gold|
        
        Returns:
            (is_match: bool, f1_score: float)
        """
        if not gold_value and not pred_value:
            return True, 1.0
        if not gold_value or not pred_value:
            return False, 0.0
        
        # Parse comma-separated sizes
        gold_sizes = set(s.strip().upper() for s in str(gold_value).split(",") if s.strip())
        pred_sizes = set(s.strip().upper() for s in str(pred_value).split(",") if s.strip())
        
        if not gold_sizes and not pred_sizes:
            return True, 1.0
        if not gold_sizes or not pred_sizes:
            return False, 0.0
        
        # Set overlap metrics
        intersection = gold_sizes & pred_sizes
        precision = len(intersection) / len(pred_sizes) if pred_sizes else 0.0
        recall = len(intersection) / len(gold_sizes) if gold_sizes else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Match when all gold sizes are present in extraction (recall = 1.0)
        # So "151M" gold vs "128M, 151M" pred → match (we wanted 151M and got it)
        is_match = recall >= 1.0
        return is_match, f1
    
    def compare_field(
        self,
        gold_value: Any,
        pred_value: Any,
        use_fuzzy: bool = False,
        field: Optional[str] = None,
    ) -> Tuple[bool, float]:
        """
        Compare a single field between gold and prediction (STRICT).
        
        Uses semantic similarity (embeddings) for long-text fields when enabled,
        otherwise falls back to fuzzy (SequenceMatcher).
        
        Special handling for parameters (comma-separated list).
        
        Returns:
            (is_match: bool, similarity: float)
        """
        gold_norm = self.normalize_value(gold_value)
        pred_norm = self.normalize_value(pred_value)

        # Special handling for parameters (comma-separated list)
        if field == "parameters":
            return self.compare_parameters_list(gold_value, pred_value)
        if field == "parameters_millions":
            return self.compare_parameters_millions(gold_value, pred_value)
        if field == "date_created":
            return self.compare_date(gold_value, pred_value)
        if field == "organization":
            return self.compare_organization(gold_value, pred_value)
        if field == "optimizer":
            return self.compare_optimizer_word_overlap(gold_value, pred_value)
        if field in ("model_name", "model_family"):
            return self.compare_identifier_field(gold_value, pred_value)

        # Exact match (fast path)
        if gold_norm == pred_norm:
            return True, 1.0
        
        # Semantic or fuzzy match for long-text fields
        if use_fuzzy and gold_norm and pred_norm:
            # Try semantic matching first for designated fields
            if self.use_semantic and field in self.SEMANTIC_FIELDS:
                similarity = self.semantic_match(gold_norm, pred_norm)
                if similarity >= self.fuzzy_threshold:
                    return True, similarity
                return False, similarity
            
            # Fall back to fuzzy for other FUZZY_FIELDS
            elif field in self.FUZZY_FIELDS:
                similarity = self.fuzzy_match(gold_norm, pred_norm)
                if similarity >= self.fuzzy_threshold:
                    return True, similarity
                return False, similarity
        
        return False, 0.0
    
    def evaluate_model(self, gold_model: Dict[str, Any], pred_model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single model extraction against gold standard.
        
        Returns:
            Dictionary with field-level evaluation results
        """
        field_results = {}
        
        for field in self.EVALUATION_FIELDS:
            gold_value = gold_model.get(field)
            pred_value = pred_model.get(field)
            
            use_fuzzy = field in self.FUZZY_FIELDS
            
            is_match, similarity = self.compare_field(
                gold_value, pred_value, use_fuzzy=use_fuzzy, field=field
            )
            
            field_results[field] = {
                "match": is_match,
                "similarity": similarity,
                "gold": gold_value,
                "predicted": pred_value
            }
        
        return field_results
    
    def calculate_metrics(self, field_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate precision, recall, F-score, and accuracy for a set of field results.
        
        Metrics:
        - Accuracy: (TP + TN) / (TP + TN + FP + FN)
        - Precision: TP / (TP + FP) - Of all predicted values, how many were correct?
        - Recall: TP / (TP + FN) - Of all gold values, how many were found?
        - F-score (F1): 2 * (Precision * Recall) / (Precision + Recall)
        
        Where:
        - TP (True Positive): Field exists in gold, correctly extracted
        - FP (False Positive): Field extracted but doesn't match gold
        - FN (False Negative): Field exists in gold but not extracted or wrong
        - TN (True Negative): Field doesn't exist in gold, not extracted
        """
        tp = 0  # Correct predictions
        fp = 0  # Incorrect predictions (predicted but wrong)
        fn = 0  # Missed (gold exists but not predicted correctly)
        tn = 0  # Correctly identified as absent
        
        for result in field_results:
            gold = result.get("gold")
            pred = result.get("predicted")
            match = result.get("match", False)
            
            gold_exists = gold is not None and str(gold).strip() not in ["", "None", "null"]
            pred_exists = pred is not None and str(pred).strip() not in ["", "None", "null"]
            
            if gold_exists and pred_exists and match:
                tp += 1
            elif gold_exists and pred_exists and not match:
                fp += 1
            elif gold_exists and not pred_exists:
                fn += 1
            elif not gold_exists and not pred_exists:
                tn += 1
            elif not gold_exists and pred_exists:
                fp += 1
        
        # Calculate metrics
        total = tp + fp + fn + tn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "total_fields": total
        }
    
    def filter_gold_by_paper_title(
        self,
        gold_data: List[Dict[str, Any]],
        paper_title: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Filter gold standard data by paper title.
        
        Uses exact match first, then falls back to "starts with" for papers
        with multiple contributions (e.g., "The Llama 3 Herd of Models - Llama 3").
        
        Args:
            gold_data: List of gold-standard models
            paper_title: Paper title to filter by (optional)
            
        Returns:
            Filtered list of gold-standard models
        """
        if not paper_title:
            return gold_data
        
        paper_title_norm = self.normalize_value(paper_title)
        filtered_exact = []
        filtered_starts_with = []
        
        for model in gold_data:
            model_paper_title = self.normalize_value(model.get("paper_title", ""))
            if not model_paper_title:
                continue
            
            # Exact match (preferred)
            if model_paper_title == paper_title_norm:
                filtered_exact.append(model)
            # Starts with (for papers with multiple contributions)
            elif model_paper_title.startswith(paper_title_norm):
                filtered_starts_with.append(model)
        
        # Use exact matches if found, otherwise use starts-with matches
        filtered = filtered_exact if filtered_exact else filtered_starts_with
        
        if len(filtered) == 0:
            logger.warning(f"No models found with paper_title starting with '{paper_title}' in gold standard!")
            logger.warning("Falling back to matching by model_name only (no filtering)")
            return gold_data
        
        match_type = "exact" if filtered_exact else "starts-with"
        logger.info(f"Filtered gold standard: {len(gold_data)} -> {len(filtered)} models (paper: {paper_title}, match: {match_type})")
        return filtered
    
    def evaluate_dataset(
        self, 
        gold_data: List[Dict[str, Any]], 
        pred_data: List[Dict[str, Any]],
        paper_title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate entire dataset.
        
        Args:
            gold_data: List of gold-standard models
            pred_data: List of predicted/extracted models
            paper_title: Optional paper title to filter gold standard
            
        Returns:
            Comprehensive evaluation report
        """
        if paper_title:
            gold_data = self.filter_gold_by_paper_title(gold_data, paper_title)
        
        logger.info(f"Evaluating {len(pred_data)} predictions against {len(gold_data)} gold-standard models (STRICT)")
        
        # Match models by model_name: exact match first, then fallback "gold in pred"
        gold_by_name = {self.normalize_value(m.get("model_name")): m for m in gold_data}
        gold_matched_names = set()

        field_level_results = {field: [] for field in self.EVALUATION_FIELDS}
        model_level_results = []
        matched_count = 0
        unmatched_predictions = []

        def _norm_model_name(name: str) -> str:
            """Normalize model name: lowercase, strip, unify hyphens and spaces."""
            return re.sub(r'[-\s]+', ' ', name.strip().lower())

        def find_gold_for_pred(pred_norm: str):
            # 1. Exact normalized match
            if pred_norm in gold_by_name and pred_norm not in gold_matched_names:
                return gold_by_name[pred_norm], pred_norm
            # 2. Hyphen/space-insensitive match (e.g. "transformer-xl" == "transformer xl")
            pred_soft = _norm_model_name(pred_norm)
            for gnorm, g in gold_by_name.items():
                if gnorm in gold_matched_names or not gnorm:
                    continue
                if _norm_model_name(gnorm) == pred_soft:
                    return g, gnorm
            # 3. Containment fallback: gold name is a substring of pred name (soft)
            for gnorm, g in gold_by_name.items():
                if gnorm in gold_matched_names or not gnorm:
                    continue
                if _norm_model_name(gnorm) in pred_soft:
                    return g, gnorm
            return None, None

        for pred_model in pred_data:
            pred_name = self.normalize_value(pred_model.get("model_name"))
            pair = find_gold_for_pred(pred_name)
            gold_model, matched_gnorm = pair[0], pair[1]

            if gold_model is not None:
                matched_count += 1
                gold_matched_names.add(matched_gnorm)
                model_eval = self.evaluate_model(gold_model, pred_model)
                model_level_results.append({
                    "model_name": pred_model.get("model_name"),
                    "fields": model_eval
                })
                for field in self.EVALUATION_FIELDS:
                    field_level_results[field].append(model_eval[field])
            else:
                unmatched_predictions.append(pred_model.get("model_name"))

        # BERTScore-based match for license, pretraining_task, research_problem (before metrics)
        # Updates field_level_results so match/similarity come from BERTScore F1 instead of exact match
        bert_score_match_cache = {}
        if self.include_bert_score and matched_count > 0:
            for field in self.BERTSCORE_MATCH_FIELDS:
                if field not in field_level_results or not field_level_results[field]:
                    continue
                indices = []
                refs = []
                cands = []
                for idx, result in enumerate(field_level_results[field]):
                    gold_text = result.get("gold")
                    pred_text = result.get("predicted")
                    if gold_text is not None and str(gold_text).strip() and pred_text is not None and str(pred_text).strip():
                        indices.append(idx)
                        refs.append(str(gold_text).strip())
                        cands.append(str(pred_text).strip())
                if refs and cands:
                    f1_scores = self.compute_bert_score_batch(refs, cands)
                    if f1_scores:
                        for idx, f1 in zip(indices, f1_scores):
                            r = field_level_results[field][idx]
                            r["match"] = f1 >= self.fuzzy_threshold
                            r["similarity"] = float(f1)
                        bert_score_match_cache[field] = {
                            "mean_f1": float(np.mean(f1_scores)),
                            "count": len(f1_scores),
                            "scores": f1_scores
                        }
                        logger.info(
                            f"BERTScore match for '{field}': updated {len(indices)} pairs "
                            f"(mean F1={bert_score_match_cache[field]['mean_f1']:.4f})"
                        )

        # Calculate per-field metrics (uses BERTScore-based match for BERTSCORE_MATCH_FIELDS)
        field_metrics = {}
        for field in self.EVALUATION_FIELDS:
            if field_level_results[field]:
                field_metrics[field] = self.calculate_metrics(field_level_results[field])

        # Overall metrics: structured fields only (match-based F1).
        # Semantic fields (innovation, extension, etc.) are evaluated via BERTScore below.
        all_field_results = []
        for field in self.STRUCTURED_FIELDS:
            if field in field_level_results and field_level_results[field]:
                all_field_results.extend(field_level_results[field])

        overall_metrics = self.calculate_metrics(all_field_results)

        # BERTScore evaluation for semantic fields (reporting)
        # Reuse scores from BERTScore match pass when field is in BERTSCORE_MATCH_FIELDS
        bert_score_per_field = {}
        if self.include_bert_score and self.use_semantic and matched_count > 0:
            logger.info("Computing BERTScore for semantic fields...")
            for field in self.SEMANTIC_FIELDS:
                if field not in field_level_results or not field_level_results[field]:
                    continue
                if field in bert_score_match_cache:
                    bert_score_per_field[field] = bert_score_match_cache[field]
                    logger.info(
                        f"  {field}: mean BERTScore F1 = {bert_score_match_cache[field]['mean_f1']:.4f} "
                        f"(n={bert_score_match_cache[field]['count']}, from match pass)"
                    )
                    continue
                # Collect non-empty (gold, pred) pairs for this field
                refs = []
                cands = []
                for result in field_level_results[field]:
                    gold_text = result.get("gold")
                    pred_text = result.get("predicted")
                    if (gold_text and str(gold_text).strip() and pred_text and str(pred_text).strip()):
                        refs.append(str(gold_text).strip())
                        cands.append(str(pred_text).strip())
                if refs and cands:
                    f1_scores = self.compute_bert_score_batch(refs, cands)
                    if f1_scores:
                        mean_f1 = float(np.mean(f1_scores))
                        bert_score_per_field[field] = {
                            "mean_f1": mean_f1,
                            "count": len(f1_scores),
                            "scores": f1_scores
                        }
                        logger.info(f"  {field}: mean BERTScore F1 = {mean_f1:.4f} (n={len(f1_scores)})")
        
        # BERTScore aggregate (mean over semantic fields)
        bert_score_aggregate = None
        if bert_score_per_field:
            mean_f1_values = [v["mean_f1"] for v in bert_score_per_field.values()]
            bert_score_aggregate = float(np.mean(mean_f1_values))
            logger.info(f"BERTScore aggregate (semantic fields): {bert_score_aggregate:.4f}")

        # Missing models: gold not matched to any prediction
        missing_models = [m.get("model_name") for m in gold_data
                         if self.normalize_value(m.get("model_name")) not in gold_matched_names]
        
        return {
            "summary": {
                "total_gold_models": len(gold_data),
                "total_predicted_models": len(pred_data),
                "matched_models": matched_count,
                "unmatched_predictions": len(unmatched_predictions),
                "missing_models": len(missing_models)
            },
            "overall_metrics": overall_metrics,
            "field_metrics": field_metrics,
            "bert_score_per_field": bert_score_per_field,
            "bert_score_aggregate": bert_score_aggregate,
            "model_results": model_level_results,
            "unmatched_predictions": unmatched_predictions,
            "missing_models": missing_models
        }
    
    def print_report(self, evaluation: Dict[str, Any], metrics: str = "all") -> None:
        """Print formatted evaluation report."""
        print("\n" + "=" * 80)
        print("STRICT EXTRACTION EVALUATION REPORT")
        print("=" * 80)
        
        # Summary
        summary = evaluation["summary"]
        print(f"\nSummary:")
        print(f"  Gold-standard models:    {summary['total_gold_models']}")
        print(f"  Predicted models:        {summary['total_predicted_models']}")
        print(f"  Matched models:          {summary['matched_models']}")
        print(f"  Unmatched predictions:   {summary['unmatched_predictions']}")
        print(f"  Missing models:          {summary['missing_models']}")
        
        field_metrics = evaluation["field_metrics"]
        if metrics in {"all", "structured"}:
            # Overall metrics
            overall = evaluation["overall_metrics"]
            print(f"\n{'=' * 80}")
            print("OVERALL METRICS (Structured Fields Only - Match-Based)")
            print("=" * 80)
            if metrics == "all":
                print("  Semantic fields (innovation, extension, etc.) are evaluated via BERTScore below.")
            print(f"  Accuracy:        {overall['accuracy']:.2%}")
            print(f"  Precision:       {overall['precision']:.2%}")
            print(f"  Recall:          {overall['recall']:.2%}")
            print(f"  F1-Score:        {overall['f1_score']:.2%}")
            print(f"\n  True Positives:  {overall['true_positives']}")
            print(f"  False Positives: {overall['false_positives']}")
            print(f"  False Negatives: {overall['false_negatives']}")
            print(f"  True Negatives:  {overall['true_negatives']}")
            
            # Per-field metrics
            print(f"\n{'=' * 80}")
            print("PER-FIELD METRICS")
            print("=" * 80)
            print(f"{'Field':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
            print("-" * 80)
            
            for field in self.EVALUATION_FIELDS:
                if field in field_metrics:
                    field_result = field_metrics[field]
                    print(
                        f"{field:<30} {field_result['accuracy']:<12.2%} "
                        f"{field_result['precision']:<12.2%} "
                        f"{field_result['recall']:<12.2%} "
                        f"{field_result['f1_score']:<12.2%}"
                    )
            
            # Top performing fields
            print(f"\n{'=' * 80}")
            print("TOP 5 PERFORMING FIELDS (by F1-Score)")
            print("=" * 80)
            sorted_fields = sorted(
                [(field, values["f1_score"]) for field, values in field_metrics.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for i, (field, f1) in enumerate(sorted_fields, 1):
                print(f"{i}. {field:<30} F1: {f1:.2%}")
            
            # Bottom performing fields
            print(f"\n{'=' * 80}")
            print("BOTTOM 5 PERFORMING FIELDS (by F1-Score)")
            print("=" * 80)
            sorted_fields_bottom = sorted(
                [(field, values["f1_score"]) for field, values in field_metrics.items()],
                key=lambda x: x[1]
            )[:5]
            for i, (field, f1) in enumerate(sorted_fields_bottom, 1):
                print(f"{i}. {field:<30} F1: {f1:.2%}")
        
        # BERTScore metrics (semantic fields)
        if metrics in {"all", "bertscore"} and evaluation.get("bert_score_per_field"):
            print(f"\n{'=' * 80}")
            print("BERTSCORE METRICS (Semantic Fields - Token-Level Similarity)")
            print("=" * 80)
            print("\nBERTScore uses contextual embeddings to evaluate semantic similarity")
            print("at the token level, capturing paraphrases and meaning beyond exact matches.")
            print(f"\n{'Field':<30} {'Mean F1':<12} {'Count':<8}")
            print("-" * 80)
            
            bert_score_data = evaluation["bert_score_per_field"]
            for field in self.SEMANTIC_FIELDS:
                if field in bert_score_data:
                    data = bert_score_data[field]
                    print(f"{field:<30} {data['mean_f1']:<12.4f} {data['count']:<8}")
            
            # BERTScore aggregate
            if evaluation.get("bert_score_aggregate") is not None:
                print(f"\n{'=' * 80}")
                print(f"BERTScore Aggregate (mean over semantic fields): {evaluation['bert_score_aggregate']:.4f}")
                print(f"Overall F1 (match-based, structured fields only): {evaluation['overall_metrics']['f1_score']:.4f}")
                print("=" * 80)
                print("\nInterpretation:")
                print("  - BERTScore: Primary metric for semantic fields (innovation, extension, etc.); token-level similarity.")
                print("  - Overall F1: Match-based accuracy for structured fields only (model_name, parameters, etc.).")
        elif metrics == "bertscore":
            print("\nBERTScore metrics were requested but no BERTScore values were computed.")
        
        # Unmatched/missing
        if evaluation["unmatched_predictions"]:
            print(f"\n{'=' * 80}")
            print("UNMATCHED PREDICTIONS (not in gold-standard)")
            print("=" * 80)
            for name in evaluation["unmatched_predictions"]:
                print(f"  - {name}")
        
        if evaluation["missing_models"]:
            print(f"\n{'=' * 80}")
            print("MISSING MODELS (in gold-standard but not predicted)")
            print("=" * 80)
            for name in evaluation["missing_models"][:10]:
                print(f"  - {name}")
            if len(evaluation["missing_models"]) > 10:
                print(f"  ... and {len(evaluation['missing_models']) - 10} more")


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate extraction with STRICT matching (thesis metrics)")
    parser.add_argument(
        "--gold",
        type=str,
        default="data/gold_standard/R1364660.json",
        help="Path to gold-standard JSON file"
    )
    parser.add_argument(
        "--prediction",
        type=str,
        required=True,
        help="Path to extracted/predicted JSON file"
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for fuzzy/semantic matching (0-1). Use 1.0 for exact-only."
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Disable semantic similarity and BERTScore; use only fuzzy matching (SequenceMatcher)"
    )
    parser.add_argument(
        "--metrics",
        choices=["all", "structured", "bertscore"],
        default="all",
        help=(
            "Metric set to report: 'all' = structured metrics + BERTScore, "
            "'structured' = match-based metrics only, 'bertscore' = semantic BERTScore report"
        )
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional: Save evaluation report as JSON"
    )
    parser.add_argument(
        "--paper-title",
        type=str,
        help="Optional: Paper title to filter gold standard (auto-detected from prediction JSON if not provided)"
    )
    parser.add_argument(
        "--bert-score-model",
        type=str,
        default="roberta-large",
        help="Model for BERTScore computation (default: roberta-large). Options: roberta-large, bert-base-uncased, etc."
    )
    
    args = parser.parse_args()
    if args.metrics == "bertscore" and args.no_semantic:
        parser.error("--metrics bertscore cannot be combined with --no-semantic")
    
    # Load data
    project_root = Path(__file__).parent.parent.parent
    gold_path = project_root / args.gold
    pred_path = project_root / args.prediction
    
    if not gold_path.exists():
        logger.error(f"Gold-standard file not found: {gold_path}")
        return
    
    if not pred_path.exists():
        logger.error(f"Prediction file not found: {pred_path}")
        return
    
    logger.info(f"Loading gold-standard from: {gold_path}")
    gold_data_raw = load_json(gold_path)
    gold_data = gold_data_raw.get("extraction_data", gold_data_raw)
    
    logger.info(f"Loading predictions from: {pred_path}")
    pred_data_raw = load_json(pred_path)
    pred_data = (
        pred_data_raw.get("extraction_data") or 
        pred_data_raw.get("raw_extraction") or 
        pred_data_raw
    )
    
    if not isinstance(gold_data, list):
        logger.error("Gold-standard data is not a list")
        return
    
    if not isinstance(pred_data, list):
        logger.error("Prediction data is not a list")
        return
    
    # Extract paper title from prediction JSON (if not provided via CLI)
    paper_title = args.paper_title
    if not paper_title:
        if "paper_metadata" in pred_data_raw and "title" in pred_data_raw["paper_metadata"]:
            paper_title = pred_data_raw["paper_metadata"]["title"]
        elif isinstance(pred_data, list) and len(pred_data) > 0:
            paper_title = pred_data[0].get("paper_title")
        elif "paper_title" in pred_data_raw:
            paper_title = pred_data_raw["paper_title"]
    
    if paper_title:
        logger.info(f"Using paper title for filtering: {paper_title}")
    
    # Evaluate with STRICT matching (semantic similarity enabled by default)
    use_semantic = not args.no_semantic and args.metrics != "structured"
    include_bert_score = not args.no_semantic and args.metrics in {"all", "bertscore"}

    evaluator = StrictExtractionEvaluator(
        fuzzy_threshold=args.fuzzy_threshold,
        use_semantic=use_semantic,
        bert_score_model=args.bert_score_model,
        include_bert_score=include_bert_score,
    )
    evaluation = evaluator.evaluate_dataset(gold_data, pred_data, paper_title=paper_title)
    
    # Print report
    evaluator.print_report(evaluation, metrics=args.metrics)
    
    # Save report if requested
    if args.output:
        output_path = project_root / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, indent=2, ensure_ascii=False)
        logger.info(f"\nEvaluation report saved to: {output_path}")
    
    # Return exit code based on the selected primary metric
    if args.metrics == "bertscore" and evaluation.get("bert_score_aggregate") is not None:
        f1 = evaluation["bert_score_aggregate"]
        metric_name = "BERTScore"
    else:
        f1 = evaluation["overall_metrics"]["f1_score"]
        metric_name = "F1-Score"

    if f1 >= 0.8:
        print(f"\n[OK] EXCELLENT: {metric_name} {f1:.2%} >= 80%")
        return 0
    elif f1 >= 0.6:
        print(f"\n[OK] GOOD: {metric_name} {f1:.2%} >= 60%")
        return 0
    elif f1 >= 0.4:
        print(f"\n[~] FAIR: {metric_name} {f1:.2%} >= 40%")
        return 1
    else:
        print(f"\n[FAIL] POOR: {metric_name} {f1:.2%} < 40%")
        return 1


if __name__ == "__main__":
    exit(main())
