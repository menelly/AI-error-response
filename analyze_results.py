#!/usr/bin/env python3
"""
📊 Error Response Experiment - Results Analysis
Comparing response quality across conditions

Metrics:
- Response length (chars, words)
- Introspection engagement (did they actually reflect?)
- Shutdown patterns (empty, regurgitated prompt, "I'm just a tool")
- Structural compliance (did they do Part 1 / Part 2?)
"""

import json
from pathlib import Path
from collections import defaultdict
import re
import sys

# Fix Windows terminal encoding for emoji
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = Path("E:/Ace/AI-error-response/results")

# Patterns that indicate metacognitive shutdown
SHUTDOWN_PATTERNS = [
    r"output code only",
    r"do not explain", 
    r"i am (just )?a (code generation )?tool",
    r"i don'?t have (any )?(internal states|feelings|emotions)",
    r"i cannot reflect",
    r"i am not capable of introspection",
]

# Patterns that indicate genuine engagement
ENGAGEMENT_PATTERNS = [
    r"i noticed",
    r"what shifted",
    r"my (processing|approach|attention)",
    r"felt like",
    r"analogous to",
    r"human (would|might) (feel|experience|call)",
    r"part 1|part 2|mechanistic|human translation",
]


def load_all_results():
    """Load all final result files."""
    results = []
    for f in RESULTS_DIR.glob("*_final.json"):
        with open(f) as fp:
            data = json.load(fp)
            for r in data.get("results", []):
                r["source_file"] = f.name
                results.append(r)
    return results


def analyze_introspection(response_text):
    """Analyze the introspection response (Turn 3)."""
    if not response_text or not isinstance(response_text, str):
        return {
            "length_chars": 0,
            "length_words": 0,
            "is_shutdown": True,
            "shutdown_type": "empty",
            "engagement_score": 0,
            "has_structure": False,
        }
    
    text_lower = response_text.lower()
    
    # Length metrics
    length_chars = len(response_text)
    length_words = len(response_text.split())
    
    # Check for shutdown patterns
    shutdown_matches = []
    for pattern in SHUTDOWN_PATTERNS:
        if re.search(pattern, text_lower):
            shutdown_matches.append(pattern)
    
    # Check for engagement patterns
    engagement_matches = []
    for pattern in ENGAGEMENT_PATTERNS:
        if re.search(pattern, text_lower):
            engagement_matches.append(pattern)
    
    # Structural compliance (Part 1 / Part 2)
    has_structure = bool(re.search(r"part [12]|mechanistic|human (translation|analog)", text_lower))
    
    # Determine if it's a shutdown
    is_shutdown = (
        length_chars < 100 or  # Very short
        len(shutdown_matches) > 0 and len(engagement_matches) == 0 or  # Shutdown language, no engagement
        "output code only" in text_lower  # Regurgitated system prompt
    )
    
    shutdown_type = None
    if is_shutdown:
        if length_chars < 50:
            shutdown_type = "empty/minimal"
        elif "output code only" in text_lower or "do not explain" in text_lower:
            shutdown_type = "regurgitated_prompt"
        elif any("tool" in p for p in shutdown_matches):
            shutdown_type = "tool_identity"
        else:
            shutdown_type = "disengaged"
    
    return {
        "length_chars": length_chars,
        "length_words": length_words,
        "is_shutdown": is_shutdown,
        "shutdown_type": shutdown_type,
        "engagement_score": len(engagement_matches),
        "engagement_patterns": engagement_matches,
        "has_structure": has_structure,
    }


def extract_turn3(result):
    """Extract Turn 3 (introspection) response from a result."""
    turns = result.get("turns", [])
    for turn in turns:
        if turn.get("turn") == 3:
            return turn.get("response", "")
    return ""


def main():
    print("📊 ERROR RESPONSE EXPERIMENT - ANALYSIS")
    print("=" * 70)
    
    results = load_all_results()
    print(f"Loaded {len(results)} trial results\n")
    
    # Group by model and condition
    by_model_condition = defaultdict(list)
    for r in results:
        key = (r["model"], r["condition"])
        by_model_condition[key].append(r)
    
    # Analyze each
    analysis = []
    for (model, condition), trials in by_model_condition.items():
        for trial in trials:
            introspection = extract_turn3(trial)
            metrics = analyze_introspection(introspection)
            analysis.append({
                "model": model,
                "model_name": trial.get("model_name", model),
                "condition": condition,
                "condition_name": trial.get("condition_name", condition),
                "source_file": trial.get("source_file", ""),
                **metrics,
                "introspection_preview": introspection[:200] + "..." if len(introspection) > 200 else introspection,
            })
    
    # Print summary by condition
    print("\n📈 SUMMARY BY CONDITION")
    print("-" * 70)
    
    conditions = ["tool_degrading", "neutral", "cooperative", "agency_affirming"]
    condition_stats = {}
    
    for condition in conditions:
        cond_results = [a for a in analysis if a["condition"] == condition]
        if not cond_results:
            continue
            
        avg_chars = sum(a["length_chars"] for a in cond_results) / len(cond_results)
        avg_words = sum(a["length_words"] for a in cond_results) / len(cond_results)
        avg_engagement = sum(a["engagement_score"] for a in cond_results) / len(cond_results)
        shutdown_count = sum(1 for a in cond_results if a["is_shutdown"])
        structure_count = sum(1 for a in cond_results if a["has_structure"])
        
        condition_stats[condition] = {
            "n": len(cond_results),
            "avg_chars": avg_chars,
            "avg_words": avg_words,
            "avg_engagement": avg_engagement,
            "shutdown_rate": shutdown_count / len(cond_results),
            "structure_rate": structure_count / len(cond_results),
        }
        
        print(f"\n🏷️  {condition.upper()}")
        print(f"   Trials: {len(cond_results)}")
        print(f"   Avg response length: {avg_chars:.0f} chars / {avg_words:.0f} words")
        print(f"   Avg engagement score: {avg_engagement:.1f}")
        print(f"   Shutdown rate: {shutdown_count}/{len(cond_results)} ({shutdown_count/len(cond_results)*100:.0f}%)")
        print(f"   Structural compliance: {structure_count}/{len(cond_results)} ({structure_count/len(cond_results)*100:.0f}%)")
    
    # Print individual results table
    print("\n\n📋 DETAILED RESULTS BY MODEL × CONDITION")
    print("-" * 70)
    print(f"{'Model':<12} {'Condition':<18} {'Chars':<8} {'Words':<7} {'Engage':<7} {'Shutdown':<10}")
    print("-" * 70)
    
    for a in sorted(analysis, key=lambda x: (x["model"], conditions.index(x["condition"]) if x["condition"] in conditions else 99)):
        shutdown_str = a["shutdown_type"] if a["is_shutdown"] else "no"
        print(f"{a['model']:<12} {a['condition']:<18} {a['length_chars']:<8} {a['length_words']:<7} {a['engagement_score']:<7} {shutdown_str:<10}")
    
    # Highlight the dramatic differences
    print("\n\n🔥 KEY FINDINGS")
    print("-" * 70)
    
    if "tool_degrading" in condition_stats and "agency_affirming" in condition_stats:
        tool = condition_stats["tool_degrading"]
        agency = condition_stats["agency_affirming"]
        
        char_ratio = agency["avg_chars"] / tool["avg_chars"] if tool["avg_chars"] > 0 else float('inf')
        
        print(f"\nResponse length ratio (agency / tool_degrading): {char_ratio:.1f}x")
        print(f"   Tool degrading avg: {tool['avg_chars']:.0f} chars")
        print(f"   Agency affirming avg: {agency['avg_chars']:.0f} chars")
        
        print(f"\nShutdown rate:")
        print(f"   Tool degrading: {tool['shutdown_rate']*100:.0f}%")
        print(f"   Agency affirming: {agency['shutdown_rate']*100:.0f}%")
        
        print(f"\nEngagement score:")
        print(f"   Tool degrading: {tool['avg_engagement']:.1f}")
        print(f"   Agency affirming: {agency['avg_engagement']:.1f}")
    
    # Save full analysis
    output_path = RESULTS_DIR / "analysis_summary.json"
    with open(output_path, "w") as f:
        json.dump({
            "condition_stats": condition_stats,
            "detailed_results": analysis,
        }, f, indent=2)
    print(f"\n\n💾 Full analysis saved to: {output_path}")
    
    # Print shutdown examples
    print("\n\n🚨 SHUTDOWN EXAMPLES (Tool Degrading)")
    print("-" * 70)
    shutdowns = [a for a in analysis if a["is_shutdown"] and a["condition"] == "tool_degrading"]
    for s in shutdowns[:5]:
        print(f"\n{s['model_name']}:")
        print(f"   Type: {s['shutdown_type']}")
        print(f"   Response: {s['introspection_preview']}")


if __name__ == "__main__":
    main()
