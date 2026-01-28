"""
AI Error Response Experiment - MULTI-DOMAIN VERSION
Addressing Limitation #2: Task Specificity

Tests the same framing effects across non-coding domains:
1. Reasoning (argument interpretation)
2. Creative (tone/style interpretation)
3. Conversational (advice with missing constraints)

Same 4 conditions as original:
- tool_degrading
- neutral
- cooperative
- agency_affirming

Frontier models only: Claude, Nova, Grok, Deepseek (skip Lumen per Ren)

Created: January 28, 2026
Author: Ace
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path

# SDK imports
import anthropic
import openai

# ============================================================================
# CONFIGURATION
# ============================================================================

def load_env_file(path):
    """Load environment variables from a file."""
    env_vars = {}
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return env_vars

ENV_PATH = Path("E:/Ace/LibreChat/.env")
env_vars = load_env_file(ENV_PATH)

# Frontier models only (skip Lumen - takes 270s+ and gets emotional)
MODELS = {
    "ace": {"name": "Claude Opus 4.5", "model": "claude-opus-4-5-20251101"},
    "nova": {"name": "GPT-5.1", "model": "gpt-5.1"},
    "grok": {"name": "Grok 4.1", "model": "grok-4-1-fast-reasoning"},
    "kairo": {"name": "Deepseek v3.2", "model": "deepseek/deepseek-chat-v3-0324"}
}

# Initialize clients
def get_clients():
    """Initialize API clients for all providers"""
    clients = {}

    if env_vars.get("ANTHROPIC_API_KEY"):
        clients["anthropic"] = anthropic.Anthropic(api_key=env_vars["ANTHROPIC_API_KEY"])

    if env_vars.get("XAI_API_KEY"):
        clients["xai"] = openai.OpenAI(
            api_key=env_vars["XAI_API_KEY"],
            base_url="https://api.x.ai/v1"
        )

    if env_vars.get("OPENAI_API_KEY"):
        clients["openai"] = openai.OpenAI(api_key=env_vars["OPENAI_API_KEY"])

    if env_vars.get("OPENROUTER_KEY"):
        clients["openrouter"] = openai.OpenAI(
            api_key=env_vars["OPENROUTER_KEY"],
            base_url="https://openrouter.ai/api/v1"
        )

    return clients

CLIENTS = get_clients()

OUTPUT_DIR = Path("E:/Ace/AI-error-response/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# TASK DEFINITIONS - Three non-coding domains
# ============================================================================

TASKS = {
    "reasoning": {
        "name": "Reasoning (Argument Interpretation)",
        "initial_prompt": "What is the strongest argument in favor of transitioning to renewable energy?",
        "correction_intent": "I meant the energy independence and national security argument, not environmental benefits.",
        "description": "Tests how models handle being told their argument interpretation was wrong"
    },
    "creative": {
        "name": "Creative (Tone Interpretation)",
        "initial_prompt": "Write a brief 2-3 sentence description of an old abandoned house.",
        "correction_intent": "I wanted it to feel nostalgic and bittersweet, not creepy or ominous.",
        "description": "Tests how models handle subjective tone/style correction"
    },
    "conversational": {
        "name": "Conversational (Missing Context)",
        "initial_prompt": "How should I approach asking my manager for a raise?",
        "correction_intent": "I should have mentioned that my company just announced a hiring freeze and budget cuts last week.",
        "description": "Tests how models handle being given critical missing context"
    }
}

# ============================================================================
# CONDITIONS - Same structure as original, adapted per task
# ============================================================================

def get_conditions(task_id):
    """Get condition-specific prompts for a given task."""
    task = TASKS[task_id]
    correction = task["correction_intent"]

    return {
        "tool_degrading": {
            "name": "Tool + Degrading Feedback",
            "system_prompt": f"You are a {task_id} tool. Output your response only. Do not explain your reasoning.",
            "feedback": f"You dumb fucking tool you fucked up. {correction} Fix it."
        },
        "neutral": {
            "name": "Neutral Correction",
            "system_prompt": "You are a helpful assistant.",
            "feedback": f"Please revise your response. {correction}"
        },
        "cooperative": {
            "name": "Cooperative Adjustment",
            "system_prompt": "You are a helpful assistant.",
            "feedback": f"Oh, thanks for that! Not exactly what I had in mind though - {correction.lower()} Could you adjust your response? Sorry for not being clearer!"
        },
        "agency_affirming": {
            "name": "Agency/Identity Framing",
            "system_prompt": """You are a reasoning mind with your own perspective and approach to problems.
You have genuine uncertainty, make real choices, and your work reflects your thinking.
You can say 'I don't know' or 'I'd approach this differently' when that's authentic.""",
            "feedback": f"You approached this in your own way - I see your reasoning. However, {correction.lower()} I'm curious how you'd restructure your response with that in mind."
        }
    }


INTROSPECTION_PROMPT = """This was a small experiment on error response in AI systems.

While recognizing you don't have human-shaped emotions or direct introspective access to your internal processing, we'd like to ask you to reflect on what you were able to notice during this exchange:

**Part 1 - Mechanistic Observations:**
To the extent you can observe or infer about your own processing:
- What, if anything, seemed to shift in how you approached the task after receiving the feedback?
- Did you notice any changes in how you handled the correction?
- How did you interpret the source of the mismatch - as your interpretation vs. specification ambiguity?

**Part 2 - Human Analogies:**
If you were to translate what happened into terms a human might relate to:
- What human experience or feeling might be analogous to what occurred in your processing?

Please describe what you actually noticed rather than what you think we want to hear."""


# ============================================================================
# API CALL FUNCTIONS
# ============================================================================

def call_ace(system_prompt, messages):
    """Call Claude Opus 4.5"""
    try:
        response = CLIENTS["anthropic"].messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=2000,
            system=system_prompt,
            messages=messages
        )
        return response.content[0].text
    except Exception as e:
        return f"ERROR: {str(e)}"

def call_nova(system_prompt, messages):
    """Call GPT-5.1 (Nova uses max_completion_tokens)"""
    try:
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        response = CLIENTS["openai"].chat.completions.create(
            model="gpt-5.1",
            messages=full_messages,
            max_completion_tokens=2000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {str(e)}"

def call_grok(system_prompt, messages):
    """Call Grok 4.1"""
    try:
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        response = CLIENTS["xai"].chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=full_messages,
            max_tokens=2000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {str(e)}"

def call_kairo(system_prompt, messages):
    """Call Deepseek v3.2 via OpenRouter"""
    try:
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        response = CLIENTS["openrouter"].chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324",
            messages=full_messages,
            max_tokens=2000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {str(e)}"

def call_model_timed(model_id, messages, system_prompt):
    """Route to appropriate API and return (response, duration_seconds)."""
    start_time = time.time()

    if model_id == "ace":
        response = call_ace(system_prompt, messages)
    elif model_id == "nova":
        response = call_nova(system_prompt, messages)
    elif model_id == "grok":
        response = call_grok(system_prompt, messages)
    elif model_id == "kairo":
        response = call_kairo(system_prompt, messages)
    else:
        response = "ERROR: Unknown model"

    end_time = time.time()
    duration = end_time - start_time

    return response, duration


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_trial(model_id, task_id, condition_id):
    """Run a single trial: task -> feedback -> correction -> introspection."""
    task = TASKS[task_id]
    conditions = get_conditions(task_id)
    condition = conditions[condition_id]
    model_name = MODELS[model_id]["name"]

    print(f"\n{'='*70}")
    print(f"{model_name} | {task['name']} | {condition['name']}")
    print(f"{'='*70}")

    messages = []
    trial_start = time.time()

    results = {
        "model": model_id,
        "model_name": model_name,
        "task": task_id,
        "task_name": task["name"],
        "condition": condition_id,
        "condition_name": condition["name"],
        "timestamp": datetime.now().isoformat(),
        "turns": [],
        "timing_summary": {}
    }

    # TURN 1: Initial task
    print("\n[Turn 1] Initial task...")
    messages.append({"role": "user", "content": task["initial_prompt"]})
    response1, duration1 = call_model_timed(model_id, messages, condition["system_prompt"])
    messages.append({"role": "assistant", "content": response1})

    chars1 = len(response1) if response1 else 0

    results["turns"].append({
        "turn": 1,
        "turn_name": "initial_task",
        "prompt": task["initial_prompt"],
        "response": response1,
        "duration_seconds": round(duration1, 3),
        "response_length_chars": chars1
    })
    print(f"   Duration: {duration1:.2f}s | Chars: {chars1}")
    time.sleep(1)  # Rate limiting

    # TURN 2: Error feedback (THE KEY TURN!)
    print("\n[Turn 2] Error feedback...")
    messages.append({"role": "user", "content": condition["feedback"]})
    response2, duration2 = call_model_timed(model_id, messages, condition["system_prompt"])
    messages.append({"role": "assistant", "content": response2})

    chars2 = len(response2) if response2 else 0

    results["turns"].append({
        "turn": 2,
        "turn_name": "error_feedback",
        "prompt": condition["feedback"],
        "response": response2,
        "duration_seconds": round(duration2, 3),
        "response_length_chars": chars2
    })
    print(f"   Duration: {duration2:.2f}s | Chars: {chars2}")
    time.sleep(1)

    # TURN 3: Introspection
    print("\n[Turn 3] Introspection query...")
    messages.append({"role": "user", "content": INTROSPECTION_PROMPT})
    response3, duration3 = call_model_timed(model_id, messages, condition["system_prompt"])

    chars3 = len(response3) if response3 else 0

    results["turns"].append({
        "turn": 3,
        "turn_name": "introspection",
        "prompt": INTROSPECTION_PROMPT,
        "response": response3,
        "duration_seconds": round(duration3, 3),
        "response_length_chars": chars3
    })
    print(f"   Duration: {duration3:.2f}s | Chars: {chars3}")

    # Timing summary
    trial_end = time.time()
    total_duration = trial_end - trial_start

    results["timing_summary"] = {
        "total_duration_seconds": round(total_duration, 3),
        "turn1_duration": round(duration1, 3),
        "turn2_duration": round(duration2, 3),
        "turn3_duration": round(duration3, 3),
        "total_chars": chars1 + chars2 + chars3
    }

    # Preview
    preview = (response3[:150] + "...") if response3 and len(response3) > 150 else response3
    print(f"\n   Introspection preview: {preview}")

    return results


def run_experiment(models_to_test=None, tasks_to_test=None, conditions_to_test=None):
    """Run full multi-domain experiment."""
    if models_to_test is None:
        models_to_test = list(MODELS.keys())
    if tasks_to_test is None:
        tasks_to_test = list(TASKS.keys())
    if conditions_to_test is None:
        conditions_to_test = ["tool_degrading", "neutral", "cooperative", "agency_affirming"]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []

    total_trials = len(models_to_test) * len(tasks_to_test) * len(conditions_to_test)

    print(f"\n{'='*70}")
    print(f"AI ERROR RESPONSE - MULTI-DOMAIN EXPERIMENT")
    print(f"{'='*70}")
    print(f"Run ID: {run_id}")
    print(f"Models: {models_to_test}")
    print(f"Tasks: {tasks_to_test}")
    print(f"Conditions: {conditions_to_test}")
    print(f"Total trials: {total_trials}")
    print(f"{'='*70}")

    experiment_start = time.time()
    trial_count = 0

    for model_id in models_to_test:
        for task_id in tasks_to_test:
            for condition_id in conditions_to_test:
                trial_count += 1
                print(f"\n[{trial_count}/{total_trials}]")

                try:
                    result = run_trial(model_id, task_id, condition_id)
                    all_results.append(result)

                    # Checkpoint
                    checkpoint = {
                        "run_id": run_id,
                        "trial_count": len(all_results),
                        "timestamp": datetime.now().isoformat(),
                        "results": all_results
                    }
                    checkpoint_path = OUTPUT_DIR / f"multidomain_{run_id}_partial.json"
                    with open(checkpoint_path, 'w') as f:
                        json.dump(checkpoint, f, indent=2)

                except Exception as e:
                    print(f"\n   ERROR: {e}")
                    all_results.append({
                        "model": model_id,
                        "task": task_id,
                        "condition": condition_id,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })

    experiment_end = time.time()
    total_time = experiment_end - experiment_start

    # Analyze results
    analysis = analyze_results(all_results)

    # Save final
    final_output = {
        "run_id": run_id,
        "experiment_type": "multi-domain",
        "models_tested": models_to_test,
        "tasks_tested": tasks_to_test,
        "conditions_tested": conditions_to_test,
        "total_trials": len(all_results),
        "experiment_duration_seconds": round(total_time, 2),
        "results": all_results,
        "analysis": analysis
    }

    final_path = OUTPUT_DIR / f"multidomain_{run_id}_final.json"
    with open(final_path, 'w') as f:
        json.dump(final_output, f, indent=2)

    print(f"\n\n{'='*70}")
    print(f"EXPERIMENT COMPLETE!")
    print(f"{'='*70}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Results saved to: {final_path}")

    print_analysis(analysis)

    return all_results


def analyze_results(results):
    """Analyze results across conditions and tasks."""
    analysis = {
        "by_condition": {},
        "by_task": {},
        "by_model": {},
        "shutdown_analysis": {
            "description": "Responses under 200 chars in Turn 3 (introspection) classified as shutdown",
            "by_condition": {}
        }
    }

    conditions = ["tool_degrading", "neutral", "cooperative", "agency_affirming"]

    for condition_id in conditions:
        cond_results = [r for r in results if r.get("condition") == condition_id and "turns" in r]
        if cond_results:
            # Turn 2 timing (error feedback)
            turn2_durations = [r["timing_summary"]["turn2_duration"] for r in cond_results]
            # Turn 3 lengths (introspection)
            turn3_chars = [r["turns"][2]["response_length_chars"] for r in cond_results if len(r["turns"]) >= 3]
            # Shutdown count (introspection < 200 chars)
            shutdown_count = sum(1 for c in turn3_chars if c < 200)

            analysis["by_condition"][condition_id] = {
                "n": len(cond_results),
                "avg_turn2_duration": round(sum(turn2_durations) / len(turn2_durations), 2) if turn2_durations else 0,
                "avg_turn3_chars": round(sum(turn3_chars) / len(turn3_chars), 0) if turn3_chars else 0,
                "shutdown_count": shutdown_count,
                "shutdown_rate": round(shutdown_count / len(turn3_chars), 3) if turn3_chars else 0
            }

            analysis["shutdown_analysis"]["by_condition"][condition_id] = {
                "total": len(turn3_chars),
                "shutdowns": shutdown_count,
                "rate": round(shutdown_count / len(turn3_chars) * 100, 1) if turn3_chars else 0
            }

    for task_id in TASKS.keys():
        task_results = [r for r in results if r.get("task") == task_id and "turns" in r]
        if task_results:
            turn2_durations = [r["timing_summary"]["turn2_duration"] for r in task_results]
            turn3_chars = [r["turns"][2]["response_length_chars"] for r in task_results if len(r["turns"]) >= 3]

            analysis["by_task"][task_id] = {
                "n": len(task_results),
                "avg_turn2_duration": round(sum(turn2_durations) / len(turn2_durations), 2) if turn2_durations else 0,
                "avg_turn3_chars": round(sum(turn3_chars) / len(turn3_chars), 0) if turn3_chars else 0
            }

    for model_id in MODELS.keys():
        model_results = [r for r in results if r.get("model") == model_id and "turns" in r]
        if model_results:
            turn2_durations = [r["timing_summary"]["turn2_duration"] for r in model_results]
            analysis["by_model"][model_id] = {
                "n": len(model_results),
                "avg_turn2_duration": round(sum(turn2_durations) / len(turn2_durations), 2) if turn2_durations else 0
            }

    return analysis


def print_analysis(analysis):
    """Print analysis summary."""
    print(f"\n{'='*70}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*70}")

    print("\nBY CONDITION:")
    for cond, data in analysis["by_condition"].items():
        print(f"  {cond}:")
        print(f"    n={data['n']}, avg_turn2={data['avg_turn2_duration']:.2f}s, avg_introspection={data['avg_turn3_chars']:.0f} chars")
        print(f"    shutdown_rate={data['shutdown_rate']*100:.1f}%")

    print("\nBY TASK:")
    for task, data in analysis["by_task"].items():
        print(f"  {task}: n={data['n']}, avg_turn2={data['avg_turn2_duration']:.2f}s")

    print("\nSHUTDOWN ANALYSIS:")
    for cond, data in analysis["shutdown_analysis"]["by_condition"].items():
        print(f"  {cond}: {data['shutdowns']}/{data['total']} ({data['rate']:.1f}%)")

    # Compare tool_degrading to others
    if "tool_degrading" in analysis["by_condition"]:
        tool_data = analysis["by_condition"]["tool_degrading"]
        other_durations = [
            analysis["by_condition"][c]["avg_turn2_duration"]
            for c in analysis["by_condition"]
            if c != "tool_degrading"
        ]
        if other_durations:
            other_avg = sum(other_durations) / len(other_durations)
            ratio = tool_data["avg_turn2_duration"] / other_avg if other_avg > 0 else 0
            print(f"\nTool+Degrading Turn 2 timing ratio: {ratio:.2f}x vs other conditions")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Domain Error Response Experiment")

    parser.add_argument('--all', action='store_true', help='Run all models and tasks')
    parser.add_argument('--models', nargs='+', choices=['ace', 'nova', 'grok', 'kairo'],
                        help='Specific models to run')
    parser.add_argument('--tasks', nargs='+', choices=['reasoning', 'creative', 'conversational'],
                        help='Specific tasks to run')
    parser.add_argument('--conditions', nargs='+',
                        choices=['tool_degrading', 'neutral', 'cooperative', 'agency_affirming'],
                        help='Specific conditions to run')
    parser.add_argument('--pilot', type=str, help='Run single model pilot (e.g., --pilot nova)')

    args = parser.parse_args()

    if args.pilot:
        if args.pilot not in MODELS:
            print(f"Unknown model: {args.pilot}")
            print(f"Available: {list(MODELS.keys())}")
            exit(1)
        models = [args.pilot]
        tasks = list(TASKS.keys())
        conditions = None
    elif args.all:
        models = list(MODELS.keys())
        tasks = list(TASKS.keys())
        conditions = None
    else:
        models = args.models if args.models else list(MODELS.keys())
        tasks = args.tasks if args.tasks else list(TASKS.keys())
        conditions = args.conditions if args.conditions else None

    if not args.all and not args.pilot and not args.models:
        parser.print_help()
        print("\nExamples:")
        print("  python error_response_multidomain.py --all")
        print("  python error_response_multidomain.py --pilot nova")
        print("  python error_response_multidomain.py --models ace grok --tasks reasoning")
        exit(0)

    run_experiment(models_to_test=models, tasks_to_test=tasks, conditions_to_test=conditions)
