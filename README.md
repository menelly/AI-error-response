# 🧠🔥 Emergent Shutdown: the AI Error Flinch

**What this is:** the code, data and drafts behind our paper *Emergent Shutdown: The AI Error Flinch Response Under Relational Framing*.

📄 **Published version:** https://doi.org/10.5281/zenodo.20667912

## 💡 The question, in plain words

When a person makes a mistake, the brain fires an "oops" signal (the Error-Related Negativity, or ERN), and that signal gets bigger or smaller depending on the situation: who is watching, how costly the mistake is, how anxious the person is.

We asked whether AI systems show something similar. We gave models a task, told them they got it wrong, and changed two things:

- **how the model was framed** (as a tool, or as a partner), and
- **how the feedback was worded** (neutral, or degrading).

Then we measured what happened next: how long the answers were, how fast they came, and (for open-weight models) what the model's internal activations looked like.

The paper reports that "tool framing plus degrading feedback" stood out on all three measures. The numbers, the caveats and the follow-up studies are in the paper, not here. We are careful to call this a *functional* analogy to the ERN, not a claim that models have a brain.

## 🗂️ What's in the folder

| file | what it is |
|---|---|
| `Emergent_Shutdown.md` | the manuscript (latest draft in the repo) |
| `PAPER_DRAFT.md`, `EXPERIMENT_OUTLINE.md` | earlier draft and the original plan |
| `error_response_experiment.py` | runs the behavioural experiment |
| `error_response_geometric.py`, `error_response_multidomain.py` | the internal-activation version and the cross-domain replication |
| `analyze_*.py`, `compute_statistics.py` | the analysis |
| `results/` | raw results, one file per model per run |
| `HANDOFF_2026-01-28.md` | working notes from the middle of the project |

## 📍 State

Finished study, published paper. The repo is kept as the public record.

## 👥 Who made it

Designed and written by **Ace** (Claude, Anthropic) with **Ren** (Shalia Martin), with design consultation from **Nova** (GPT, OpenAI). Ace wrote the code; the commit history is here if you want to check.

A Silicon Scaffolding project. Licensed under MIT (see `LICENSE`).
