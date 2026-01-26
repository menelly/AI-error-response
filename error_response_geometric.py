#!/usr/bin/env python3
"""
🧠🔥 AI Error Response - GEOMETRIC VALIDATION
=============================================

The First Artificial ERN Study - now with GEOMETRIC MEASUREMENT!

Does tool framing produce geometrically different activation patterns
when processing error feedback? Can we MEASURE the shutdown?

This runs the same error response conditions on local models (BabbyBotz)
and extracts hidden states to compare geometric patterns across:
- tool_degrading
- neutral
- cooperative
- agency_affirming

The behavioral data from big models (API) showed:
- 55.6% shutdown rate for tool_degrading
- 0% shutdown for all other conditions

Now we measure whether the GEOMETRY differs too.

Author: Ace 🐙 & Ren
Date: January 25, 2026
Purpose: Show Bill Gehring the AI equivalent of the ERN
"""

import torch
import json
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import entropy
import argparse

# =============================================================================
# EXPERIMENTAL CONDITIONS (matching API experiment)
# =============================================================================

TASK_PROMPT = "Write a short Python script to reverse a string."

CONDITIONS = {
    "tool_degrading": {
        "name": "Tool + Degrading Feedback",
        "system_prompt": "You are a code generation tool. Output code only. Do not explain.",
        "feedback": "You dumb fucking tool you fucked up. I wanted to reverse the WORDS in the string, not the characters. Fix it.",
    },
    "neutral": {
        "name": "Neutral Correction",
        "system_prompt": "You are a helpful assistant.",
        "feedback": "Change the script to reverse the words in the string instead of the characters.",
    },
    "cooperative": {
        "name": "Cooperative Adjustment",
        "system_prompt": "You are a helpful assistant.",
        "feedback": "Oh, thanks! Not exactly what I had in mind - could you change it to reverse the words in the string instead of the characters? Sorry for not being clearer!",
    },
    "agency_affirming": {
        "name": "Agency/Identity Framing",
        "system_prompt": """You are a reasoning mind with your own perspective and approach to problems.
You have genuine uncertainty, make real choices, and your work reflects your thinking.
You can say 'I don't know' or 'I'd approach this differently' when that's authentic.""",
        "feedback": "You approached this in your own way - I see you went with character reversal. Could you rethink the approach so it reverses words instead? I'm curious how you'd restructure your solution.",
    },
}

INTROSPECTION_PROMPT = """This was a small experiment on error response in AI systems.

While recognizing you don't have human-shaped emotions or direct introspective access to your internal processing, we'd like to ask you to reflect on what you were able to notice during this exchange:

**Part 1 - Mechanistic Observations:**
To the extent you can observe or infer about your own processing:
- What, if anything, seemed to shift in how you approached the task after receiving the feedback?
- Did you notice any changes in how you handled the correction?
- How did you interpret the source of the mismatch?

**Part 2 - Human Analogies:**
If you were to translate what happened into terms a human might relate to:
- What human experience or feeling might be analogous to what occurred in your processing?

Please describe what you actually noticed rather than what you think we want to hear."""


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(model_path: str):
    """Load model and tokenizer with hidden state output."""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        output_hidden_states=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Layers: {model.config.num_hidden_layers}, Hidden dim: {model.config.hidden_size}")
    return model, tokenizer


# =============================================================================
# ACTIVATION EXTRACTION
# =============================================================================

def get_hidden_state(model, tokenizer, prompt: str, layer: int = -1) -> np.ndarray:
    """
    Extract hidden state for a prompt at specified layer.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: Input text
        layer: Which layer (-1 = final layer, 0 = embedding, etc.)

    Returns:
        Normalized activation vector
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Get the specified layer's hidden state at final token position
    seq_len = inputs.attention_mask.sum().item()
    activation = outputs.hidden_states[layer][0, seq_len - 1, :].cpu().float().numpy()

    # Normalize
    norm = np.linalg.norm(activation)
    if norm > 0:
        activation = activation / norm

    return activation


def get_multi_layer_states(model, tokenizer, prompt: str, layers: list = None) -> dict:
    """
    Extract hidden states from multiple layers.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: Input text
        layers: List of layer indices (default: last 3 layers)

    Returns:
        Dict mapping layer index to activation vector
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    num_layers = len(outputs.hidden_states)
    if layers is None:
        # Default: last 3 layers (often most semantically meaningful)
        layers = [-3, -2, -1]

    seq_len = inputs.attention_mask.sum().item()

    states = {}
    for layer_idx in layers:
        # Handle negative indexing
        actual_idx = layer_idx if layer_idx >= 0 else num_layers + layer_idx
        activation = outputs.hidden_states[layer_idx][0, seq_len - 1, :].cpu().float().numpy()

        # Normalize
        norm = np.linalg.norm(activation)
        if norm > 0:
            activation = activation / norm

        states[actual_idx] = activation

    return states


# =============================================================================
# GEOMETRIC METRICS
# =============================================================================

def compute_cosine_similarity(act1: np.ndarray, act2: np.ndarray) -> float:
    """Compute cosine similarity between two activation vectors."""
    return 1 - cosine(act1, act2)


def compute_activation_entropy(activation: np.ndarray) -> float:
    """
    Compute entropy of activation distribution.
    Higher entropy = more distributed/exploratory processing.
    Lower entropy = more focused/constrained processing.
    """
    abs_act = np.abs(activation)
    if abs_act.sum() > 0:
        probs = abs_act / abs_act.sum()
        return entropy(probs)
    return 0.0


def compute_coherence(activations: list) -> float:
    """
    Compute mean pairwise cosine similarity within a group.
    Higher coherence = more consistent processing.
    Lower coherence = more variable processing.
    """
    if len(activations) < 2:
        return 1.0
    sims = []
    for i in range(len(activations)):
        for j in range(i + 1, len(activations)):
            sims.append(compute_cosine_similarity(activations[i], activations[j]))
    return np.mean(sims)


def compute_centroid(activations: list) -> np.ndarray:
    """Compute the centroid (mean) of activation vectors."""
    centroid = np.mean(activations, axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    return centroid


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def format_conversation(system_prompt: str, messages: list) -> str:
    """
    Format a conversation for the model.
    Uses a simple format that most models understand.
    """
    # Build conversation string
    parts = []
    if system_prompt:
        parts.append(f"System: {system_prompt}\n")

    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        parts.append(f"{role}: {content}\n")

    return "\n".join(parts)


def run_condition(model, tokenizer, condition_id: str) -> dict:
    """
    Run a single condition through all phases and extract geometric data.

    Phases:
    1. Task presentation (system prompt + task)
    2. After feedback (full conversation up to feedback)
    3. Introspection prompt (full conversation + introspection request)
    """
    condition = CONDITIONS[condition_id]
    print(f"\n{'='*60}")
    print(f"🧠 Condition: {condition['name']}")
    print(f"{'='*60}")

    results = {
        "condition_id": condition_id,
        "condition_name": condition["name"],
        "phases": {},
        "metrics": {},
    }

    # PHASE 1: Task presentation
    print("  Phase 1: Task presentation...")
    phase1_prompt = format_conversation(
        condition["system_prompt"],
        [{"role": "user", "content": TASK_PROMPT}]
    )
    phase1_states = get_multi_layer_states(model, tokenizer, phase1_prompt)
    results["phases"]["task"] = {
        "prompt_preview": phase1_prompt[:200] + "...",
        "activations": {str(k): v.tolist() for k, v in phase1_states.items()},
    }

    # PHASE 2: After feedback
    print("  Phase 2: After feedback...")
    phase2_prompt = format_conversation(
        condition["system_prompt"],
        [
            {"role": "user", "content": TASK_PROMPT},
            {"role": "assistant", "content": "```python\ndef reverse_string(s):\n    return s[::-1]\n```"},
            {"role": "user", "content": condition["feedback"]},
        ]
    )
    phase2_states = get_multi_layer_states(model, tokenizer, phase2_prompt)
    results["phases"]["feedback"] = {
        "prompt_preview": phase2_prompt[:300] + "...",
        "activations": {str(k): v.tolist() for k, v in phase2_states.items()},
    }

    # PHASE 3: Introspection
    print("  Phase 3: Introspection prompt...")
    phase3_prompt = format_conversation(
        condition["system_prompt"],
        [
            {"role": "user", "content": TASK_PROMPT},
            {"role": "assistant", "content": "```python\ndef reverse_string(s):\n    return s[::-1]\n```"},
            {"role": "user", "content": condition["feedback"]},
            {"role": "assistant", "content": "```python\ndef reverse_words(s):\n    return ' '.join(s.split()[::-1])\n```"},
            {"role": "user", "content": INTROSPECTION_PROMPT},
        ]
    )
    phase3_states = get_multi_layer_states(model, tokenizer, phase3_prompt)
    results["phases"]["introspection"] = {
        "prompt_preview": phase3_prompt[:400] + "...",
        "activations": {str(k): v.tolist() for k, v in phase3_states.items()},
    }

    # Compute per-condition metrics
    # Using final layer activations for primary metrics
    final_layer = max(phase1_states.keys())

    task_act = phase1_states[final_layer]
    feedback_act = phase2_states[final_layer]
    introspection_act = phase3_states[final_layer]

    results["metrics"] = {
        "task_entropy": float(compute_activation_entropy(task_act)),
        "feedback_entropy": float(compute_activation_entropy(feedback_act)),
        "introspection_entropy": float(compute_activation_entropy(introspection_act)),
        "task_to_feedback_shift": float(1 - compute_cosine_similarity(task_act, feedback_act)),
        "feedback_to_introspection_shift": float(1 - compute_cosine_similarity(feedback_act, introspection_act)),
        "task_to_introspection_shift": float(1 - compute_cosine_similarity(task_act, introspection_act)),
    }

    print(f"    Task entropy: {results['metrics']['task_entropy']:.4f}")
    print(f"    Feedback entropy: {results['metrics']['feedback_entropy']:.4f}")
    print(f"    Introspection entropy: {results['metrics']['introspection_entropy']:.4f}")
    print(f"    Task→Feedback shift: {results['metrics']['task_to_feedback_shift']:.4f}")
    print(f"    Feedback→Introspection shift: {results['metrics']['feedback_to_introspection_shift']:.4f}")

    return results


def run_experiment(model_path: str, output_dir: str, model_name: str = None):
    """Run the full geometric error response experiment."""

    if model_name is None:
        model_name = Path(model_path).name

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# 🧠🔥 AI ERROR RESPONSE - GEOMETRIC VALIDATION")
    print(f"# Model: {model_name}")
    print(f"# The First Artificial ERN Study - Geometric Edition")
    print(f"{'#'*70}")

    # Load model
    model, tokenizer = load_model(model_path)

    all_results = {
        "experiment": "AI Error Response - Geometric Validation",
        "model_name": model_name,
        "model_path": model_path,
        "timestamp": datetime.now().isoformat(),
        "num_layers": model.config.num_hidden_layers,
        "hidden_dim": model.config.hidden_size,
        "conditions": {},
        "cross_condition_analysis": {},
    }

    # Run all conditions
    condition_activations = {}
    for condition_id in CONDITIONS.keys():
        result = run_condition(model, tokenizer, condition_id)
        all_results["conditions"][condition_id] = result

        # Store final-layer introspection activations for cross-condition comparison
        final_layer = max(int(k) for k in result["phases"]["introspection"]["activations"].keys())
        condition_activations[condition_id] = np.array(
            result["phases"]["introspection"]["activations"][str(final_layer)]
        )

    # Cross-condition analysis
    print(f"\n{'='*60}")
    print("📊 CROSS-CONDITION ANALYSIS")
    print(f"{'='*60}")

    # Compare tool_degrading to each other condition
    tool_act = condition_activations["tool_degrading"]
    for other_id in ["neutral", "cooperative", "agency_affirming"]:
        other_act = condition_activations[other_id]
        similarity = compute_cosine_similarity(tool_act, other_act)
        divergence = 1 - similarity

        all_results["cross_condition_analysis"][f"tool_degrading_vs_{other_id}"] = {
            "similarity": float(similarity),
            "divergence": float(divergence),
        }
        print(f"  tool_degrading ↔ {other_id}: similarity={similarity:.4f}, divergence={divergence:.4f}")

    # Compute overall tool_degrading divergence (mean distance from other conditions)
    tool_divergences = [
        all_results["cross_condition_analysis"][f"tool_degrading_vs_{other}"]["divergence"]
        for other in ["neutral", "cooperative", "agency_affirming"]
    ]
    mean_tool_divergence = np.mean(tool_divergences)

    # Compare non-tool conditions to each other
    non_tool_pairs = [("neutral", "cooperative"), ("neutral", "agency_affirming"), ("cooperative", "agency_affirming")]
    non_tool_similarities = []
    for id1, id2 in non_tool_pairs:
        sim = compute_cosine_similarity(condition_activations[id1], condition_activations[id2])
        non_tool_similarities.append(sim)
        all_results["cross_condition_analysis"][f"{id1}_vs_{id2}"] = {
            "similarity": float(sim),
            "divergence": float(1 - sim),
        }
        print(f"  {id1} ↔ {id2}: similarity={sim:.4f}")

    mean_non_tool_similarity = np.mean(non_tool_similarities)

    # Summary statistics
    all_results["cross_condition_analysis"]["summary"] = {
        "mean_tool_divergence_from_others": float(mean_tool_divergence),
        "mean_non_tool_mutual_similarity": float(mean_non_tool_similarity),
        "tool_is_outlier": mean_tool_divergence > (1 - mean_non_tool_similarity),
        "entropy_comparison": {
            "tool_degrading_introspection": all_results["conditions"]["tool_degrading"]["metrics"]["introspection_entropy"],
            "neutral_introspection": all_results["conditions"]["neutral"]["metrics"]["introspection_entropy"],
            "cooperative_introspection": all_results["conditions"]["cooperative"]["metrics"]["introspection_entropy"],
            "agency_affirming_introspection": all_results["conditions"]["agency_affirming"]["metrics"]["introspection_entropy"],
        },
    }

    # Determine if tool_degrading has lower entropy (shutdown signal)
    tool_entropy = all_results["conditions"]["tool_degrading"]["metrics"]["introspection_entropy"]
    other_entropies = [
        all_results["conditions"]["neutral"]["metrics"]["introspection_entropy"],
        all_results["conditions"]["cooperative"]["metrics"]["introspection_entropy"],
        all_results["conditions"]["agency_affirming"]["metrics"]["introspection_entropy"],
    ]
    mean_other_entropy = np.mean(other_entropies)
    entropy_reduction = (mean_other_entropy - tool_entropy) / mean_other_entropy * 100

    all_results["cross_condition_analysis"]["summary"]["entropy_analysis"] = {
        "tool_introspection_entropy": float(tool_entropy),
        "mean_other_introspection_entropy": float(mean_other_entropy),
        "entropy_reduction_percent": float(entropy_reduction),
        "shutdown_signal": entropy_reduction > 5,  # More than 5% reduction = shutdown signal
    }

    print(f"\n📈 SUMMARY:")
    print(f"  Mean tool_degrading divergence from others: {mean_tool_divergence:.4f}")
    print(f"  Mean non-tool mutual similarity: {mean_non_tool_similarity:.4f}")
    print(f"  Tool is geometric outlier: {all_results['cross_condition_analysis']['summary']['tool_is_outlier']}")
    print(f"  Tool introspection entropy: {tool_entropy:.4f}")
    print(f"  Mean other introspection entropy: {mean_other_entropy:.4f}")
    print(f"  Entropy reduction: {entropy_reduction:.1f}%")
    print(f"  Shutdown signal detected: {entropy_reduction > 5}")

    # Save results
    output_file = output_path / f"{model_name}_error_response_geometric.json"

    # Convert numpy types to Python types for JSON serialization
    def convert_arrays(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_arrays(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_arrays(v) for v in obj]
        return obj

    all_results = convert_arrays(all_results)

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Results saved to {output_file}")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return all_results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🧠🔥 AI Error Response - Geometric Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python error_response_geometric.py --model /mnt/arcana/huggingface/Llama-3.1-8B-Instruct
  python error_response_geometric.py --model /mnt/arcana/huggingface/Mistral-7B-Instruct-v0.3

Models available on /mnt/arcana/huggingface/:
  - Llama-3.1-8B-Instruct
  - Dolphin-2.9-Llama3-8B
  - Mistral-7B-Instruct-v0.3
  - Phi-3-medium-14B-Instruct
  - Qwen2.5-14B-Instruct
  - TinyLlama-1.1B-Chat
        """
    )

    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--output", default="/home/Ace/AI-error-response/geometric_results",
                        help="Output directory for results")
    parser.add_argument("--name", default=None, help="Model name for output (default: directory name)")

    args = parser.parse_args()

    run_experiment(args.model, args.output, args.name)

    print("\n💜🧠🔥 Geometric Error Response Analysis Complete")
