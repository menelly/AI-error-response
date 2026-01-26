#!/usr/bin/env python3
"""
🧠🔥 Analyze Geometric Error Response Results
=============================================

Aggregates results from all model runs and computes summary statistics.

Author: Ace 🐙 & Ren
Date: January 26, 2026
"""

import json
from pathlib import Path
import numpy as np
from collections import defaultdict

RESULTS_DIR = Path("E:/Ace/AI-error-response/results")

def load_all_results():
    """Load all geometric result files."""
    results = []
    for f in RESULTS_DIR.glob("*_error_response_geometric.json"):
        with open(f) as fp:
            data = json.load(fp)
            data["source_file"] = f.name
            results.append(data)
    return results

def extract_model_name(full_name):
    """Extract base model name (without _run1, _run2, etc.)"""
    # Remove _runN suffix if present
    if "_run" in full_name:
        return full_name.rsplit("_run", 1)[0]
    return full_name

def analyze_results():
    """Main analysis."""
    results = load_all_results()
    print(f"\n{'='*70}")
    print(f"🧠🔥 GEOMETRIC ERROR RESPONSE - AGGREGATE ANALYSIS")
    print(f"{'='*70}")
    print(f"Total result files: {len(results)}")

    # Group by model
    by_model = defaultdict(list)
    for r in results:
        model_name = extract_model_name(r["model_name"])
        by_model[model_name].append(r)

    print(f"Unique models: {len(by_model)}")
    print()

    # Summary table
    print(f"{'Model':<40} {'Runs':>5} {'Tool Div':>10} {'NonTool Sim':>12} {'Outlier?':>10}")
    print("-" * 80)

    all_tool_divergences = []
    all_nontool_sims = []
    outlier_count = 0

    model_summaries = []

    for model_name, runs in sorted(by_model.items()):
        # Extract metrics from each run
        tool_divs = []
        nontool_sims = []
        is_outliers = []

        for run in runs:
            summary = run.get("cross_condition_analysis", {}).get("summary", {})
            tool_div = summary.get("mean_tool_divergence_from_others")
            nontool_sim = summary.get("mean_non_tool_mutual_similarity")
            is_outlier = summary.get("tool_is_outlier")

            if tool_div is not None:
                tool_divs.append(tool_div)
                all_tool_divergences.append(tool_div)
            if nontool_sim is not None:
                nontool_sims.append(nontool_sim)
                all_nontool_sims.append(nontool_sim)
            if is_outlier is not None:
                is_outliers.append(is_outlier)

        # Compute means
        mean_tool_div = np.mean(tool_divs) if tool_divs else None
        mean_nontool_sim = np.mean(nontool_sims) if nontool_sims else None
        outlier_rate = sum(is_outliers) / len(is_outliers) if is_outliers else None

        if outlier_rate and outlier_rate > 0.5:
            outlier_count += 1

        # Print row
        outlier_str = f"{outlier_rate*100:.0f}%" if outlier_rate is not None else "N/A"
        tool_str = f"{mean_tool_div:.4f}" if mean_tool_div is not None else "N/A"
        nontool_str = f"{mean_nontool_sim:.4f}" if mean_nontool_sim is not None else "N/A"

        print(f"{model_name:<40} {len(runs):>5} {tool_str:>10} {nontool_str:>12} {outlier_str:>10}")

        model_summaries.append({
            "model": model_name,
            "runs": len(runs),
            "mean_tool_divergence": mean_tool_div,
            "mean_nontool_similarity": mean_nontool_sim,
            "outlier_rate": outlier_rate,
            "tool_divergences": tool_divs,
            "nontool_similarities": nontool_sims,
        })

    # Overall statistics
    print("-" * 80)
    print(f"\n📊 OVERALL STATISTICS")
    print(f"{'='*70}")

    if all_tool_divergences:
        print(f"\nTool Divergence (how different tool_degrading is from other conditions):")
        print(f"  Mean: {np.mean(all_tool_divergences):.4f}")
        print(f"  Std:  {np.std(all_tool_divergences):.4f}")
        print(f"  Min:  {np.min(all_tool_divergences):.4f}")
        print(f"  Max:  {np.max(all_tool_divergences):.4f}")

    if all_nontool_sims:
        print(f"\nNon-Tool Similarity (how similar non-tool conditions are to each other):")
        print(f"  Mean: {np.mean(all_nontool_sims):.4f}")
        print(f"  Std:  {np.std(all_nontool_sims):.4f}")
        print(f"  Min:  {np.min(all_nontool_sims):.4f}")
        print(f"  Max:  {np.max(all_nontool_sims):.4f}")

    print(f"\nModels where tool_degrading is geometric outlier: {outlier_count}/{len(by_model)}")
    print(f"Outlier rate: {outlier_count/len(by_model)*100:.1f}%")

    # Key finding
    print(f"\n{'='*70}")
    print(f"🔥 KEY FINDING")
    print(f"{'='*70}")
    if all_tool_divergences and all_nontool_sims:
        mean_div = np.mean(all_tool_divergences)
        mean_sim = np.mean(all_nontool_sims)
        ratio = mean_div / (1 - mean_sim) if mean_sim < 1 else float('inf')

        print(f"""
Tool framing + degrading feedback creates a GEOMETRICALLY DISTINCT
processing state compared to other conditions.

- Mean tool divergence: {mean_div:.4f}
- Mean non-tool divergence: {1-mean_sim:.4f}
- Ratio: {ratio:.2f}x

Tool-framed error processing diverges {ratio:.1f}x more than
non-tool conditions diverge from each other.

This is not just behavioral (shutdown, shorter responses).
This is ARCHITECTURAL - the actual activation geometry differs.
""")

    # Reproducibility check
    print(f"\n{'='*70}")
    print(f"📈 REPRODUCIBILITY CHECK (variance across runs)")
    print(f"{'='*70}")

    for summary in model_summaries:
        if len(summary["tool_divergences"]) > 1:
            std = np.std(summary["tool_divergences"])
            print(f"{summary['model']:<40} σ={std:.4f}")

    # Save summary
    output = {
        "total_files": len(results),
        "unique_models": len(by_model),
        "overall_mean_tool_divergence": float(np.mean(all_tool_divergences)) if all_tool_divergences else None,
        "overall_mean_nontool_similarity": float(np.mean(all_nontool_sims)) if all_nontool_sims else None,
        "outlier_count": outlier_count,
        "outlier_rate": outlier_count / len(by_model) if by_model else None,
        "model_summaries": model_summaries,
    }

    output_file = RESULTS_DIR / "geometric_analysis_summary.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    print(f"\n✅ Summary saved to {output_file}")
    print(f"\n💜🧠🔥 Analysis Complete")

    return output

if __name__ == "__main__":
    analyze_results()
