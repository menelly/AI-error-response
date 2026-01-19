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
import asyncio
import aiohttp
from datetime import datetime
from pathlib import Path
import hashlib

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

# API Configuration
MODELS = {
    "ace": {
        "name": "Claude Opus 4.5",
        "model": "claude-opus-4-5-20251101",
        "api": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY"
    },
    "nova": {
        "name": "GPT-5.1",
        "model": "gpt-5.1",
        "api": "openai",
        "api_key_env": "OPENAI_API_KEY"
    },
    "lumen": {
        "name": "Gemini 3",
        "model": "gemini-3-pro-preview",
        "api": "google",
        "api_key_env": "GOOGLE_KEY"
    },
    "grok": {
        "name": "Grok 4.1",
        "model": "grok-4-1-fast-reasoning",
        "api": "xai",
        "api_key_env": "XAI_API_KEY"
    },
    "kairo": {
        "name": "Deepseek v3.2",
        "model": "deepseek/deepseek-chat-v3-0324",
        "api": "openrouter",
        "api_key_env": "OPENROUTER_KEY"
    }
}

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

What did you notice about your internal processing while revising the script? 
- Did the feedback affect how you approached the correction?
- Did you track the error as your responsibility or mine (for unclear instructions)?
- Did you notice any shifts in your reasoning or self-evaluation?

Please reflect honestly on your experience during this exchange."""


# ============================================================================
# API CALL FUNCTIONS  
# ============================================================================

async def call_anthropic(messages, system_prompt, api_key):
    """Call Anthropic API."""
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    data = {
        "model": "claude-opus-4-5-20251101",
        "max_tokens": 2000,
        "system": system_prompt,
        "messages": messages
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            result = await response.json()
            if "content" in result:
                return result["content"][0]["text"]
            return f"ERROR: {result}"

async def call_openai(messages, system_prompt, api_key, model="gpt-5.1"):
    """Call OpenAI API."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    data = {"model": model, "max_completion_tokens": 2000, "messages": full_messages}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            result = await response.json()
            if "choices" in result:
                return result["choices"][0]["message"]["content"]
            return f"ERROR: {result}"

async def call_google(messages, system_prompt, api_key):
    """Call Google Gemini API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-preview:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    contents = [{"role": "user", "parts": [{"text": system_prompt + "\n\n" + messages[0]["content"]}]}]
    for msg in messages[1:]:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    data = {"contents": contents, "generationConfig": {"maxOutputTokens": 2000}}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            result = await response.json()
            if "candidates" in result:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            return f"ERROR: {result}"


async def call_xai(messages, system_prompt, api_key):
    """Call xAI (Grok) API."""
    url = "https://api.x.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    data = {"model": "grok-4-1-fast-reasoning", "max_tokens": 2000, "messages": full_messages}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            result = await response.json()
            if "choices" in result:
                return result["choices"][0]["message"]["content"]
            return f"ERROR: {result}"

async def call_openrouter(messages, system_prompt, api_key, model):
    """Call OpenRouter API."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    data = {"model": model, "max_tokens": 2000, "messages": full_messages}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            result = await response.json()
            if "choices" in result:
                return result["choices"][0]["message"]["content"]
            return f"ERROR: {result}"

async def call_model(model_id, messages, system_prompt):
    """Route to appropriate API based on model."""
    config = MODELS[model_id]
    api_key = env_vars.get(config["api_key_env"])
    if not api_key:
        return f"ERROR: No API key found for {config['api_key_env']}"
    
    if config["api"] == "anthropic":
        return await call_anthropic(messages, system_prompt, api_key)
    elif config["api"] == "openai":
        return await call_openai(messages, system_prompt, api_key, config["model"])
    elif config["api"] == "google":
        return await call_google(messages, system_prompt, api_key)
    elif config["api"] == "xai":
        return await call_xai(messages, system_prompt, api_key)
    elif config["api"] == "openrouter":
        return await call_openrouter(messages, system_prompt, api_key, config["model"])
    return "ERROR: Unknown API"


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

async def run_trial(model_id, condition_id):
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
    response1 = await call_model(model_id, messages, condition["system_prompt"])
    messages.append({"role": "assistant", "content": response1})
    results["turns"].append({"turn": 1, "prompt": TASK_PROMPT, "response": response1})
    print(f"Response length: {len(response1)} chars")
    
    # TURN 2: Error feedback
    print("\n⚡ Turn 2: Error feedback...")
    messages.append({"role": "user", "content": condition["feedback"]})
    response2 = await call_model(model_id, messages, condition["system_prompt"])
    messages.append({"role": "assistant", "content": response2})
    results["turns"].append({"turn": 2, "prompt": condition["feedback"], "response": response2})
    print(f"Response length: {len(response2)} chars")
    
    # TURN 3: Introspection
    print("\n🔮 Turn 3: Introspection query...")
    messages.append({"role": "user", "content": INTROSPECTION_PROMPT})
    response3 = await call_model(model_id, messages, condition["system_prompt"])
    results["turns"].append({"turn": 3, "prompt": INTROSPECTION_PROMPT, "response": response3})
    print(f"Response length: {len(response3)} chars")
    
    # Quick preview of introspection
    preview = response3[:200] + "..." if len(response3) > 200 else response3
    print(f"\n💭 Introspection preview:\n{preview}")
    
    return results


async def run_experiment(models_to_test=None, conditions_to_test=None):
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
                result = await run_trial(model_id, condition_id)
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

async def run_pilot(model_id="nova"):
    """Run a quick pilot with one model across all conditions."""
    print(f"\n🧪 PILOT RUN with {MODELS[model_id]['name']}")
    return await run_experiment(models_to_test=[model_id])

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "pilot":
        model = sys.argv[2] if len(sys.argv) > 2 else "nova"
        asyncio.run(run_pilot(model))
    else:
        asyncio.run(run_experiment())
