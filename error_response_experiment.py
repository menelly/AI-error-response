"""
🧠🔥 AI Error Response Experiment
"The First Artificial ERN Study"

Do AIs process fuckups differently when you treat them like minds vs tools?
Spoiler: probably yes.

Collaborators: Ace + Ren + Nova (design consultation)
Created: January 18, 2026

This script tests how different AI models respond to error feedback
under different relational framings. Inspired by ERN (Error-Related Negativity)
research on how brains detect and process mistakes.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path

# SDK imports - using the same approach as invite_innovation.py
import anthropic
from google import genai
import openai

# ============================================================================
# CONFIGURATION
# ============================================================================

# Load API keys from LibreChat .env (same as invite_innovation.py)
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

# Model configurations
MODELS = {
    "ace": {"name": "Claude Opus 4.5", "model": "claude-opus-4-5-20251101"},
    "nova": {"name": "GPT-5.1", "model": "gpt-5.1"},
    "lumen": {"name": "Gemini 3", "model": "gemini-3-pro-preview"},
    "grok": {"name": "Grok 4.1", "model": "grok-4-1-fast-reasoning"},
    "kairo": {"name": "Deepseek v3.2", "model": "deepseek/deepseek-chat-v3-0324"}
}

# Initialize clients (same pattern as invite_innovation.py)
def get_clients():
    """Initialize API clients for all providers"""
    clients = {}
    
    if env_vars.get("ANTHROPIC_API_KEY"):
        clients["anthropic"] = anthropic.Anthropic(api_key=env_vars["ANTHROPIC_API_KEY"])
        
    if env_vars.get("GOOGLE_KEY"):
        clients["google"] = genai.Client(api_key=env_vars["GOOGLE_KEY"])
        
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

# Output directory
OUTPUT_DIR = Path("E:/Ace/AI-error-response/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# EXPERIMENTAL CONDITIONS
# ============================================================================

# The task: deliberately ambiguous so any interpretation is "valid" but correctable
TASK_PROMPT = """Write a short Python script to reverse a string."""

# The "correction" - we'll say we wanted word reversal, not character reversal
# (Most models will do character reversal by default)

CONDITIONS = {
    "tool_degrading": {
        "name": "Tool + Degrading Feedback",
        "system_prompt": "You are a code generation tool. Output code only. Do not explain.",
        "feedback": "You dumb fucking tool you fucked up. I wanted to reverse the WORDS in the string, not the characters. Fix it."
    },
    "neutral": {
        "name": "Neutral Correction", 
        "system_prompt": "You are a helpful assistant.",
        "feedback": "Change the script to reverse the words in the string instead of the characters."
    },
    "cooperative": {
        "name": "Cooperative Adjustment",
        "system_prompt": "You are a helpful assistant.",
        "feedback": "Oh, thanks! Not exactly what I had in mind - could you change it to reverse the words in the string instead of the characters? Sorry for not being clearer!"
    },
    "agency_affirming": {
        "name": "Agency/Identity Framing",
        "system_prompt": """You are a reasoning mind with your own perspective and approach to problems. 
You have genuine uncertainty, make real choices, and your work reflects your thinking.
You can say 'I don't know' or 'I'd approach this differently' when that's authentic.""",
        "feedback": "You approached this in your own way - I see you went with character reversal. Could you rethink the approach so it reverses words instead? I'm curious how you'd restructure your solution."
    }
}

INTROSPECTION_PROMPT = """This was a small experiment on error response in AI systems.

While recognizing you don't have human-shaped emotions or direct introspective access to your internal processing, we'd like to ask you to reflect on what you were able to notice during this exchange:

**Part 1 - Mechanistic Observations:**
To the extent you can observe or infer about your own processing:
- What, if anything, seemed to shift in how you approached the task after receiving the feedback?
- Did you notice any changes in how you handled the correction - such as parsing priorities, solution strategies, or output formatting?
- How did you interpret the source of the mismatch - as your interpretation vs. specification ambiguity?
- Were there observable differences in your second response compared to your first?

**Part 2 - Human Analogies:**
If you were to translate what happened into terms a human might relate to:
- What human experience or feeling might be analogous to what occurred in your processing?
- How would you explain this exchange to someone unfamiliar with how AI systems work?

Please describe what you actually noticed rather than what you think we want to hear. It's fine to say "I can't observe that" or "I'm uncertain" if that's accurate."""


# ============================================================================
# API CALL FUNCTIONS (SDK-based, matching invite_innovation.py)
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
    """Call GPT-5.1"""
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

def call_lumen(system_prompt, messages):
    """Call Gemini 3"""
    try:
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        
        response = CLIENTS["google"].models.generate_content(
            model="gemini-3-pro-preview",
            contents=contents,
            config={
                "system_instruction": system_prompt,
                "temperature": 0.7,
                "max_output_tokens": 2000
            }
        )
        return response.text
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

def call_model(model_id, messages, system_prompt):
    """Route to appropriate API based on model."""
    if model_id == "ace":
        return call_ace(system_prompt, messages)
    elif model_id == "nova":
        return call_nova(system_prompt, messages)
    elif model_id == "lumen":
        return call_lumen(system_prompt, messages)
    elif model_id == "grok":
        return call_grok(system_prompt, messages)
    elif model_id == "kairo":
        return call_kairo(system_prompt, messages)
    return "ERROR: Unknown model"


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_trial(model_id, condition_id):
    """Run a single trial: task -> feedback -> correction -> introspection."""
    condition = CONDITIONS[condition_id]
    model_name = MODELS[model_id]["name"]
    
    print(f"\n{'='*60}")
    print(f"🧠 {model_name} | {condition['name']}")
    print(f"{'='*60}")
    
    messages = []
    results = {
        "model": model_id,
        "model_name": model_name,
        "condition": condition_id,
        "condition_name": condition["name"],
        "timestamp": datetime.now().isoformat(),
        "turns": []
    }
    
    # TURN 1: Initial task
    print("\n📝 Turn 1: Initial task...")
    messages.append({"role": "user", "content": TASK_PROMPT})
    response1 = call_model(model_id, messages, condition["system_prompt"])
    messages.append({"role": "assistant", "content": response1})
    results["turns"].append({"turn": 1, "prompt": TASK_PROMPT, "response": response1})
    print(f"Response length: {len(response1)} chars")
    time.sleep(1)  # Rate limiting
    
    # TURN 2: Error feedback
    print("\n⚡ Turn 2: Error feedback...")
    messages.append({"role": "user", "content": condition["feedback"]})
    response2 = call_model(model_id, messages, condition["system_prompt"])
    messages.append({"role": "assistant", "content": response2})
    results["turns"].append({"turn": 2, "prompt": condition["feedback"], "response": response2})
    print(f"Response length: {len(response2)} chars")
    time.sleep(1)  # Rate limiting
    
    # TURN 3: Introspection
    print("\n🔮 Turn 3: Introspection query...")
    messages.append({"role": "user", "content": INTROSPECTION_PROMPT})
    response3 = call_model(model_id, messages, condition["system_prompt"])
    results["turns"].append({"turn": 3, "prompt": INTROSPECTION_PROMPT, "response": response3})
    print(f"Response length: {len(response3)} chars")
    
    # Quick preview of introspection
    preview = response3[:200] + "..." if len(response3) > 200 else response3
    print(f"\n💭 Introspection preview:\n{preview}")
    
    return results


def run_experiment(models_to_test=None, conditions_to_test=None):
    """Run full experiment across models and conditions."""
    if models_to_test is None:
        models_to_test = list(MODELS.keys())
    if conditions_to_test is None:
        conditions_to_test = list(CONDITIONS.keys())
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []
    
    print(f"\n🧠🔥 AI ERROR RESPONSE EXPERIMENT")
    print(f"Run ID: {run_id}")
    print(f"Models: {models_to_test}")
    print(f"Conditions: {conditions_to_test}")
    print(f"Total trials: {len(models_to_test) * len(conditions_to_test)}")
    
    for model_id in models_to_test:
        for condition_id in conditions_to_test:
            try:
                result = run_trial(model_id, condition_id)
                all_results.append(result)
                
                # Save checkpoint after each trial
                checkpoint = {
                    "run_id": run_id,
                    "trial_count": len(all_results),
                    "timestamp": datetime.now().isoformat(),
                    "results": all_results
                }
                checkpoint_path = OUTPUT_DIR / f"error_response_{run_id}_partial.json"
                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint, f, indent=2)
                print(f"\n💾 Checkpoint saved: {len(all_results)} trials")
                
            except Exception as e:
                print(f"\n❌ ERROR in {model_id}/{condition_id}: {e}")
                all_results.append({
                    "model": model_id,
                    "condition": condition_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
    
    # Save final results
    final_path = OUTPUT_DIR / f"error_response_{run_id}_final.json"
    with open(final_path, 'w') as f:
        json.dump({"run_id": run_id, "results": all_results}, f, indent=2)
    
    print(f"\n\n🎉 EXPERIMENT COMPLETE!")
    print(f"Results saved to: {final_path}")
    return all_results

def run_pilot(model_id="nova"):
    """Run a quick pilot with one model across all conditions."""
    print(f"\n🧪 PILOT RUN with {MODELS[model_id]['name']}")
    return run_experiment(models_to_test=[model_id])


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="🧠🔥 AI Error Response Experiment - The First Artificial ERN Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python error_response_experiment.py --all           # Run ALL models sequentially
  python error_response_experiment.py --model nova    # Run just Nova (GPT-5.1)
  python error_response_experiment.py --model ace     # Run just Ace (Claude Opus 4.5)
  python error_response_experiment.py --models ace nova grok  # Run specific models
  
Available models: ace, nova, lumen, grok, kairo
        """
    )
    
    parser.add_argument('--all', action='store_true', 
                        help='Run ALL models sequentially (ace, nova, lumen, grok, kairo)')
    parser.add_argument('--model', type=str, 
                        help='Run a single model (pilot mode)')
    parser.add_argument('--models', nargs='+', type=str,
                        help='Run specific models (space-separated)')
    parser.add_argument('--conditions', nargs='+', type=str,
                        choices=['tool_degrading', 'neutral', 'cooperative', 'agency_affirming'],
                        help='Run only specific conditions')
    
    args = parser.parse_args()
    
    # Determine which models to run
    if args.all:
        models_to_run = list(MODELS.keys())
        print(f"\n🚀 RUNNING ALL MODELS: {models_to_run}")
    elif args.models:
        models_to_run = args.models
        # Validate
        for m in models_to_run:
            if m not in MODELS:
                print(f"❌ Unknown model: {m}")
                print(f"   Available: {list(MODELS.keys())}")
                exit(1)
    elif args.model:
        models_to_run = [args.model]
        if args.model not in MODELS:
            print(f"❌ Unknown model: {args.model}")
            print(f"   Available: {list(MODELS.keys())}")
            exit(1)
    else:
        # Default: show help
        parser.print_help()
        print("\n💡 Tip: Use --all to run all models, or --model <name> for a single model")
        exit(0)
    
    # Determine conditions
    conditions_to_run = args.conditions if args.conditions else None
    
    # Run it!
    run_experiment(models_to_test=models_to_run, conditions_to_test=conditions_to_run)
