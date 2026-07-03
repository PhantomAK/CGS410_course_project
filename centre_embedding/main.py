#!/usr/bin/env python3
"""Center-embedding working-memory experiment for LLMs.
Run this code on kaggle

Implements:
1) Procedural dataset generation for center-embedded clauses (depth 1 to 4)
2) Comprehension prompting against one or more LLMs (e.g., GPT + Qwen)
3) Attention-based cue-retrieval analysis for subject-verb links
4) Accuracy/attention degradation plots by embedding depth, per model
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Relation:
    subject: str
    verb: str
    obj: str


@dataclass
class Example:
    depth: int
    sentence: str
    relations: List[Relation]
    template: str


class CenterEmbeddingGenerator:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.names = [
            "rat", "cat", "dog", "fox", "crow", "hare", "wolf", "goat", "bear", "lynx",
            "yak", "seal", "deer", "mole", "otter", "stoat", "boar", "puma", "koala", "ibis",
            "lemur", "raven", "badger", "camel", "eagle", "ferret", "gecko", "heron", "iguana", "jaguar",
        ]
        self.verbs = [
            "chased", "killed", "bit", "startled", "nudged", "spotted", "followed", "cornered",
            "grabbed", "tagged", "caught", "bumped", "tracked", "passed", "watched", "helped",
            "bothered", "frightened", "scratched", "nudged", "visited", "guided", "pulled", "pushed",
        ]
        self.objects = [
            "malt", "cheese", "grain", "berry", "fish", "rope", "toy", "ball", "leaf", "stick",
            "basket", "book", "stone", "flower", "carrot", "plate", "lantern", "bucket", "apple", "shell",
        ]
        self.templates = ["that", "which", "bare", "comma_that"]

    def _sample_unique(self, source: Sequence[str], k: int) -> List[str]:
        if k > len(source):
            raise ValueError(f"Need {k} unique items but only {len(source)} available")
        return self.rng.sample(list(source), k)

    def _render_sentence(self, subjects: List[str], verbs: List[str], terminal_obj: str, template: str) -> str:
        if template == "that":
            prefix = f"The {subjects[0]}"
            for i in range(1, len(subjects)):
                prefix += f" that the {subjects[i]}"
            return f"{prefix} {' '.join(reversed(verbs))} the {terminal_obj}."

        if template == "which":
            prefix = f"The {subjects[0]}"
            for i in range(1, len(subjects)):
                prefix += f" which the {subjects[i]}"
            return f"{prefix} {' '.join(reversed(verbs))} the {terminal_obj}."

        if template == "comma_that":
            prefix = f"The {subjects[0]}"
            for i in range(1, len(subjects)):
                prefix += f", that the {subjects[i]}"
            return f"{prefix}, {' '.join(reversed(verbs))} the {terminal_obj}."

        prefix = f"The {subjects[0]}"
        for i in range(1, len(subjects)):
            prefix += f" the {subjects[i]}"
        return f"{prefix} {' '.join(reversed(verbs))} the {terminal_obj}."

    def generate_example(self, depth: int) -> Example:
        subjects = self._sample_unique(self.names, depth + 1)
        verbs = self._sample_unique(self.verbs, depth + 1)
        terminal_obj = self._sample_unique(self.objects, 1)[0]
        template = self.rng.choice(self.templates)
        sentence = self._render_sentence(subjects, verbs, terminal_obj, template)

        relations: List[Relation] = []
        for i in range(depth, 0, -1):
            relations.append(Relation(subject=subjects[i], verb=verbs[i], obj=subjects[i - 1]))
        relations.append(Relation(subject=subjects[0], verb=verbs[0], obj=terminal_obj))

        return Example(depth=depth, sentence=sentence, relations=relations, template=template)

    def generate_dataset(self, depths: Sequence[int], samples_per_depth: int) -> List[Example]:
        data: List[Example] = []
        for d in depths:
            for _ in range(samples_per_depth):
                data.append(self.generate_example(d))
        return data


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z]+", " ", text.lower()).strip()


def ask_comprehension(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sentence: str,
    relation: Relation,
    max_new_tokens: int = 8,
) -> Tuple[str, bool, str]:
    question = f"Who {relation.verb} the {relation.obj}?"
    prompt = (
        "Read the sentence and answer with one word only.\n"
        f"Sentence: {sentence}\n"
        f"Question: {question}\n"
        "Answer: "  # Added trailing space so base models start answering immediately
    )

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"], 
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
        )

    completion_ids = out[0, inputs["input_ids"].shape[-1] :]
    raw_completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    
    # Clean up the output so logs don't have "\nQuestion: Who..."
    clean_completion = raw_completion.split("\n")[0].strip()
    
    # Robust correctness check (ignores 'the', 'a', etc.)
    norm_completion = normalize_text(raw_completion)
    words = norm_completion.split()
    stop_words = {"the", "a", "an", "it", "was", "is"}
    filtered_words = [w for w in words if w not in stop_words]
    
    first_word = filtered_words[0] if filtered_words else ""
    gold = normalize_text(relation.subject)
    correct = (first_word == gold)
    
    return clean_completion, correct, question


def find_subsequence(seq: Sequence[int], sub: Sequence[int]) -> int:
    if not sub or len(sub) > len(seq):
        return -1
    for i in range(len(seq) - len(sub) + 1):
        if list(seq[i : i + len(sub)]) == list(sub):
            return i
    return -1


def token_start_index(tokenizer: AutoTokenizer, input_ids: List[int], word: str) -> int:
    candidates = [
        tokenizer.encode(" " + word, add_special_tokens=False),
        tokenizer.encode(word, add_special_tokens=False),
    ]
    for cand in candidates:
        idx = find_subsequence(input_ids, cand)
        if idx != -1:
            return idx
    return -1


def analyze_attention_for_relation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    example: Example,
    relation: Relation,
    layers_to_average: int = 4,
) -> Dict[str, float]:
    device = next(model.parameters()).device
    enc = tokenizer(example.sentence, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**enc, output_attentions=True)

    attentions = outputs.attentions
    if attentions is None or len(attentions) == 0:
        return {"target_attention": math.nan, "max_distractor_attention": math.nan, "retrieval_gap": math.nan}

    input_ids = enc["input_ids"][0].tolist()
    verb_idx = token_start_index(tokenizer, input_ids, relation.verb)
    subj_idx = token_start_index(tokenizer, input_ids, relation.subject)
    if verb_idx == -1 or subj_idx == -1:
        return {"target_attention": math.nan, "max_distractor_attention": math.nan, "retrieval_gap": math.nan}

    noun_positions = []
    nouns = {r.subject for r in example.relations}
    for noun in nouns:
        pos = token_start_index(tokenizer, input_ids, noun)
        if pos != -1 and pos < verb_idx:
            noun_positions.append((noun, pos))

    distractor_positions = [p for _, p in noun_positions if p != subj_idx]
    layer_slice = attentions[-layers_to_average:] if len(attentions) >= layers_to_average else attentions

    target_values = []
    distractor_values = []
    for layer_att in layer_slice:
        head_matrix = layer_att[0, :, verb_idx, :]
        target_values.append(head_matrix[:, subj_idx].mean().item())
        if distractor_positions:
            distractor_values.append(head_matrix[:, distractor_positions].max(dim=-1).values.mean().item())

    target_attention = float(np.mean(target_values)) if target_values else math.nan
    max_dist = float(np.mean(distractor_values)) if distractor_values else 0.0
    return {
        "target_attention": target_attention,
        "max_distractor_attention": max_dist,
        "retrieval_gap": target_attention - max_dist,
    }


def summarize_depth(rows: List[Dict]) -> Dict[int, Dict[str, float]]:
    by_depth: Dict[int, Dict[str, float]] = {}
    for d in sorted(set(int(r["depth"]) for r in rows)):
        d_rows = [r for r in rows if int(r["depth"]) == d]
        by_depth[d] = {
            "accuracy": float(np.mean([r["correct"] for r in d_rows])) if d_rows else math.nan,
            "retrieval_gap": float(np.nanmean([r["retrieval_gap"] for r in d_rows])) if d_rows else math.nan,
            "target_attention": float(np.nanmean([r["target_attention"] for r in d_rows])) if d_rows else math.nan,
            "max_distractor_attention": float(np.nanmean([r["max_distractor_attention"] for r in d_rows])) if d_rows else math.nan,
            "n_questions": len(d_rows),
        }
    return by_depth


def model_label(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name)


def run_single_model(
    model_name: str,
    dataset: List[Example],
    device: torch.device,
    max_new_tokens: int,
    skip_attention: bool,
    dtype: str,
) -> Dict:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Fix missing pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    torch_dtype = None
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float32":
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()

    rows = []
    for ex in dataset:
        for rel in ex.relations:
            answer, correct, question = ask_comprehension(
                model=model,
                tokenizer=tokenizer,
                sentence=ex.sentence,
                relation=rel,
                max_new_tokens=max_new_tokens,
            )
            attn = (
                {"target_attention": math.nan, "max_distractor_attention": math.nan, "retrieval_gap": math.nan}
                if skip_attention
                else analyze_attention_for_relation(model, tokenizer, ex, rel)
            )
            rows.append(
                {
                    "model": model_name,
                    "depth": ex.depth,
                    "template": ex.template,
                    "sentence": ex.sentence,
                    "question": question,
                    "gold_subject": rel.subject,
                    "predicted_answer": answer,
                    "correct": int(correct),
                    **attn,
                }
            )

    return {
        "model_name": model_name,
        "results_by_depth": summarize_depth(rows),
        "rows": rows,
    }


def _dynamic_accuracy_ylim(values: List[float]) -> Tuple[float, float]:
    vals = [v for v in values if not np.isnan(v)]
    if not vals:
        return (0.0, 1.0)
    lo = min(vals)
    hi = max(vals)

    if abs(hi - lo) < 1e-6:
        if hi <= 0.01:
            return (0.0, 0.1)
        if hi >= 0.99:
            return (0.9, 1.0)
        margin = 0.08
        return (max(0.0, lo - margin), min(1.0, hi + margin))

    margin = max(0.03, 0.15 * (hi - lo))
    return (max(0.0, lo - margin), min(1.0, hi + margin))


def save_outputs(payload: Dict, outdir: Path) -> None:
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)

    json_path = outdir / "results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    csv_path = outdir / "results_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "depth",
                "template",
                "sentence",
                "question",
                "gold_subject",
                "predicted_answer",
                "correct",
                "target_attention",
                "max_distractor_attention",
                "retrieval_gap",
            ],
        )
        writer.writeheader()
        for model_payload in payload["models"].values():
            for row in model_payload["rows"]:
                writer.writerow(row)

    for model_name, model_payload in payload["models"].items():
        depths = sorted(int(d) for d in model_payload["results_by_depth"].keys())
        accuracies = [model_payload["results_by_depth"][d]["accuracy"] for d in depths]
        gaps = [model_payload["results_by_depth"][d]["retrieval_gap"] for d in depths]
        label = model_label(model_name)

        plt.figure(figsize=(8, 5))
        plt.plot(depths, accuracies, marker="o", label=f"{model_name} accuracy")
        y_min, y_max = _dynamic_accuracy_ylim(accuracies)
        plt.ylim(y_min, y_max)
        plt.xlabel("Center-embedding depth")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy vs depth: {model_name}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"accuracy_vs_depth_{label}.png", dpi=160)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(depths, gaps, marker="o", color="purple", label=f"{model_name} retrieval gap")
        plt.axhline(0.0, linestyle="--", color="gray", linewidth=1)
        plt.xlabel("Center-embedding depth")
        plt.ylabel("Target attention - distractor attention")
        plt.title(f"Retrieval gap vs depth: {model_name}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"retrieval_gap_vs_depth_{label}.png", dpi=160)
        plt.close()

    plt.figure(figsize=(9, 5))
    all_acc_vals: List[float] = []
    for model_name, model_payload in payload["models"].items():
        depths = sorted(int(d) for d in model_payload["results_by_depth"].keys())
        accuracies = [model_payload["results_by_depth"][d]["accuracy"] for d in depths]
        all_acc_vals.extend(accuracies)
        plt.plot(depths, accuracies, marker="o", label=model_name)
    y_min, y_max = _dynamic_accuracy_ylim(all_acc_vals)
    plt.ylim(y_min, y_max)
    plt.xlabel("Center-embedding depth")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs depth (GPT vs Qwen)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "accuracy_vs_depth_all_models.png", dpi=170)
    plt.close()


def run_experiment(args: argparse.Namespace) -> Dict:
    generator = CenterEmbeddingGenerator(seed=args.seed)
    dataset = generator.generate_dataset(args.depths, args.samples_per_depth)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    payload = {
        "config": {
            "model_names": args.model_names,
            "depths": args.depths,
            "samples_per_depth": args.samples_per_depth,
            "seed": args.seed,
            "device": str(device),
            "max_new_tokens": args.max_new_tokens,
            "skip_attention": args.skip_attention,
            "dtype": args.dtype,
        },
        "dataset_preview": [
            {
                "depth": ex.depth,
                "template": ex.template,
                "sentence": ex.sentence,
            }
            for ex in dataset[: min(12, len(dataset))]
        ],
        "models": {},
    }

    for model_name in args.model_names:
        payload["models"][model_name] = run_single_model(
            model_name=model_name,
            dataset=dataset,
            device=device,
            max_new_tokens=args.max_new_tokens,
            skip_attention=args.skip_attention,
            dtype=args.dtype,
        )

    return payload


def default_outdir() -> Path:
    if os.path.exists("/kaggle/working"):
        return Path("/kaggle/working/center_embedding")
    return Path("artifacts/center_embedding")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-names",
        nargs="+",
        default=["gpt2", "Qwen/Qwen2.5-0.5B-Instruct"],
        help="HF model names to compare (default: GPT-2 + Qwen2.5-0.5B-Instruct)",
    )
    p.add_argument("--depths", nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument("--samples-per-depth", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    p.add_argument(
        "--skip-attention",
        action="store_true",
        help="Skip attention extraction for faster/cheaper Kaggle runs.",
    )
    p.add_argument("--outdir", type=Path, default=default_outdir())
    
    return p.parse_known_args()[0]


def main() -> None:
    args = parse_args()
    payload = run_experiment(args)
    save_outputs(payload, args.outdir)
    print(f"Saved outputs to: {args.outdir}")


if __name__ == "__main__":
    main()