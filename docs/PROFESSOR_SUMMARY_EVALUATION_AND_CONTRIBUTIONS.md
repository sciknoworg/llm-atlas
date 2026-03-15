# Summary: Evaluation and Contribution Pipeline

**Purpose:** This document summarizes how the LLM extraction pipeline is evaluated against the gold standard, how model variants are merged into single contributions, and how we decide which extracted entries count as separate contributions. It is intended for thesis progress reporting.

---

## 1. Pipeline Overview: What Happens After Extraction

After the LLM extracts a list of "models" from a paper, three steps are applied so the result is clean and comparable to the gold standard:

| Step | What it does | Example |
|------|----------------|--------|
| **1. Normalize** | Unify date and organization formats | "2018" → "2018-01", "Google AI" → "Google" |
| **2. Select primary contributions** | Remove entries that are not the main model (e.g. adapters, safety tools) | Keep "BERT", "GPT-2"; remove "Llama Guard", "BERT adapter" |
| **3. Merge variants** | Combine all size/variant versions of the same model into one contribution | "GPT-2 124M" + "GPT-2 1.5B" → one "GPT-2" with parameters "124M, 1.5B" |

Evaluation is then run on this final list against the gold standard.

---

## 2. How We Evaluate (Strict Evaluator)

We use a **strict evaluator**: we first match each predicted model to at most one gold model by **model name** (after normalizing). Then we compare field-by-field only for those matched pairs. Below, "we" = our pipeline output; "gold" = gold standard.

---

### 2.1 Model name and model family

| What we do | Normalize (lowercase, unify spaces/hyphens), then accept exact match, or one string containing the other, or high text similarity (e.g. ≥ 80%). |
| Why | Names are written differently in papers and in the gold standard. |

**Examples:**

| Gold | We | Result |
|------|-----|--------|
| "Transformer-XL" | "Transformer XL" | Match (same after normalizing) |
| "BERT" | "BERT Base" | Match (gold contained in ours) |
| "GPT-2" | "GPT-3" | No match |

---

### 2.2 Date created

| What we do | Extract (year, month). Match if same year; if both have month, we require same month. |
| Why | Gold might have "2018", we might have "2018-10-01"; we want to count that as correct. |

**Examples:**

| Gold | We | Result |
|------|-----|--------|
| "2018" | "2018-10" | Match (same year) |
| "2018-10" | "2018-10-01" | Match (same year and month) |
| "2018-06" | "2018-10" | No match (different months) |

---

### 2.3 Organization

| What we do | Exact match, or one string contains the other, or known aliases (e.g. "Google AI Language" = "Google", "Meta AI" = "Meta"). |
| Why | Same institution appears under different names. |

**Examples:**

| Gold | We | Result |
|------|-----|--------|
| "Google" | "Google AI Language" | Match (alias) |
| "Meta" | "Facebook AI" | Match (alias) |
| "OpenAI" | "Microsoft" | No match |

---

### 2.4 Parameters (sizes, e.g. 124M, 1.5B)

| What we do | Split both by commas into sets of sizes. We count a **match** when every size in the gold set appears in our set (recall = 100%). We still compute precision/recall for the metric. |
| Why | A paper may list several sizes; we want to credit "we got all the sizes the gold has"; we allow extra sizes. |

**Examples:**

| Gold | We | Result |
|------|-----|--------|
| "124M, 355M, 1.5B" | "124M, 355M, 774M, 1.5B" | Match (all gold sizes in ours) |
| "1.5B" | "124M, 1.5B" | Match (gold's only size in ours) |
| "124M, 1.5B" | "124M" | No match (we miss 1.5B) |

---

### 2.5 Parameters in millions (single number)

| What we do | Compare the numeric value; if both are null/empty, we count as match. |
| Why | One number per contribution; we need exact agreement. |

**Examples:**

| Gold | We | Result |
|------|-----|--------|
| 1500 | 1500 | Match |
| 117 | 117 | Match |
| 1500 | 774 | No match |

---

### 2.6 Optimizer

| What we do | Split both into words; if any word in our answer appears in the gold (e.g. "Adam" in "Adam optimizer"), we count match. |
| Why | We care that the right optimizer is identified, not the exact phrase. |

**Examples:**

| Gold | We | Result |
|------|-----|--------|
| "Adam optimizer" | "Adam" | Match |
| "AdamW" | "AdamW" | Match |
| "Adam" | "SGD" | No match |

---

### 2.7 Long-text fields (innovation, pretraining corpus, application, etc.)

| What we do | We use **BERTScore** (token-level semantic similarity): if BERTScore F1 ≥ 0.8 we count a match. We also report mean BERTScore per field. Fallback: fuzzy or semantic similarity with threshold. |
| Why | Wording varies a lot; we want to reward "same meaning," not exact text. |

**Examples:**

| Gold | We | Result |
|------|-----|--------|
| "BERT uses masked language modeling to pre-train bidirectional representations." | "Masked language model (MLM) pre-training for bidirectional context." | Match if BERTScore says they are similar enough (same idea, different words) |
| "Causal language modeling for next-token prediction." | "Pre-trained on next token prediction." | Match when similarity above threshold |

---

### 2.8 Other fields (architecture, tasks, hardware, license, etc.)

| What we do | For short/categorical fields we use exact match after normalization; where configured, long text uses the same fuzzy/semantic logic as above. |
| Examples | Gold "Encoder" / We "Encoder" → Match. Gold "Decoder" / We "encoder" → Match after lowercasing if we normalize. |

---

## 3. How We Merge Variants (One Contribution per Model)

**Goal:** The gold standard has **one row per model** (e.g. one "GPT-2" with all sizes listed). We do the same: we merge all "variants" of the same model into one contribution.

### 3.1 Step 1: Canonical name (what counts as "same model")

We take the model name and strip parts that only describe **size**, **context**, or **training stage**, but **keep version numbers** so that e.g. "Llama 3" and "Llama 3.1" stay different.

**Examples of canonical names:**

| Input name | Canonical name |
|------------|----------------|
| "GPT-2 1.5B" | "GPT-2" |
| "BERT Base" | "BERT" |
| "Llama 3.1 8B" | "Llama 3.1" (version 3.1 kept) |
| "Llama 3 8K-context" | "Llama 3" |
| "Llama 3 (pre-trained)" | "Llama 3" |
| "Transformer-XL 151M (ablation)" | "Transformer-XL" |
| "XLNet-Base-wikibooks" | "XLNet" |

So **"Llama 3"** and **"Llama 3.1"** remain **two different** contributions; but **"Llama 3 8B"** and **"Llama 3 70B"** become **one** contribution: **"Llama 3"**.

### 3.2 Step 2: Group and merge

All extracted entries that share the same canonical name are grouped. For each group:

- **If there is only one entry:** we only set its `model_name` to the canonical name (e.g. "Llama 3.1 405B" → name "Llama 3.1", parameters "405B").
- **If there are several** (e.g. GPT-2 124M, 355M, 774M, 1.5B): we create **one** contribution with:
  - **model_name**: canonical name (e.g. "GPT-2").
  - **parameters**: all sizes in one comma-separated list, sorted (e.g. "124M, 355M, 774M, 1.5B").
  - **parameters_millions**: the **maximum** over the variants (e.g. 1500 for 1.5B).
  - Other fields (innovation, organization, etc.): we merge by rules (e.g. prefer non-empty, for long text take the longest, for lists merge and deduplicate).

**Example of merging:**  
Extraction gives: "GPT-2 124M", "GPT-2 355M", "GPT-2 1.5B".  
After merging we have **one** contribution: model_name = "GPT-2", parameters = "124M, 355M, 1.5B", and other fields merged from the three rows.

---

## 4. How We Create Separate Contributions (and Remove Non-Contributions)

We want **one contribution per main model** (after merging), and we do **not** want adapters, tools, or classifiers to count as separate contributions.

### 4.1 Step 1: Contribution selector (remove non-models)

We drop entries that look **auxiliary** (e.g. adapter, chat, classifier, detector, embedding, filter, guard, reward model, safety, tokenizer, tool). We also use a "dominant model family" and "does it look like a real release?" (e.g. has version or size info, or many fields filled). If we are not confident, we **do not** remove anything (we keep all).

**Examples:**

| Keep | Remove |
|------|--------|
| "BERT", "GPT-2", "Llama 3.1" | "Llama Guard" (safety), "BERT adapter", "reward model" |

### 4.2 Step 2: Merging (see Section 3)

After selection, we merge variants by canonical name. So:

- **Separate contributions** = one per canonical model (e.g. "BERT", "GPT-2", "Llama 3", "Llama 3.1").
- **One contribution** can list **multiple sizes** in the `parameters` field (e.g. "124M, 355M, 1.5B").

**Example for one paper:**  
Paper introduces Llama 3 in 8B and 70B, plus "Llama Guard".  
After selection we keep "Llama 3 8B", "Llama 3 70B" and drop "Llama Guard".  
After merging we have **one** contribution "Llama 3" with parameters "8B, 70B".  
So we end up with **one** contribution for this paper's main model, matching the gold-standard idea of "one row per model."

---

## 5. Short Recap

| Topic | Summary |
|-------|---------|
| **Evaluation** | We match predicted models to gold by name, then compare each field with rules that fit the field (dates: year/month; org: aliases; parameters: set of sizes; long text: BERTScore). That gives a strict but fair metric for the thesis. |
| **Merging** | We define a "canonical model name" by stripping size/context/stage/corpus but keeping version, group by it, and merge into one contribution per group with combined sizes and merged other fields. |
| **Separate contributions** | We first remove auxiliary entries (adapters, tools, etc.), then merge variants; the result is one contribution per main model, with possible multiple sizes in one row, aligned with the gold-standard structure. |

---

## 6. Technical References (for reproducibility)

| Component | Location in codebase |
|-----------|----------------------|
| Strict evaluator | `scripts/evaluation/evaluate_extraction_strict.py` |
| Model variant merger | `src/model_variant_merger.py` |
| Contribution selector | `src/model_contribution_selector.py` |
| Extraction normalizer | `src/extraction_normalizer.py` |
| Pipeline order | `src/pipeline.py` (normalize → select → merge) |
| Evaluation README | `scripts/evaluation/README.md` |

---

*Document generated for thesis progress reporting. Last updated: February 2026.*
