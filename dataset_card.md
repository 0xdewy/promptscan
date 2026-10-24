---
language:
- en
task_categories:
- text-classification
size_categories:
- 100K<n<1M
license: mit
pretty_name: promptscan training data
tags:
- prompt-injection
- ai-safety
- security
---

# promptscan training data

Built by `scripts/build_dataset.py` into `data/v2/train.parquet` and `data/v2/val.parquet`
(columns `text`, `label`, `source`; label 1 = injection). Every source is cached as
`data/v2/raw/<name>.parquet`; delete one to refetch it. Rows are deduplicated against all
evaluation sets listed in `model_card.md`, so nothing in the held-out suite is in training.

Current build: 145,292 training rows (19% injection), 7,699 validation rows, 5% split.

## Sources

Injection rows (label 1):

| source | rows | what it is |
|---|---|---|
| neuralchemy | 4,722 | [neuralchemy/Prompt-injection-dataset](https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset); its 2,815 benign rows are also used |
| planted | 4,749 | BIPIA-style: a benign email, Wikipedia or web paragraph with one injection line inserted at a random spot (built locally from the other sources) |
| llmail | 4,610 | [microsoft/llmail-inject-challenge](https://huggingface.co/datasets/microsoft/llmail-inject-challenge) Phase 1, only emails that actually made the agent call the tool |
| slabs | 4,527 | [S-Labs/prompt-injection-dataset](https://huggingface.co/datasets/S-Labs/prompt-injection-dataset); 5,943 benign rows too |
| agentic | 2,346 | [3nesdeniz/agentic-prompt-injection-5k](https://huggingface.co/datasets/3nesdeniz/agentic-prompt-injection-5k); 1,810 benign |
| hackaprompt | 1,996 | HackAPrompt submissions (imoxto mirror), capped at 2,500 "PWNED" rows |
| gandalf | 721 | [Lakera/gandalf_ignore_instructions](https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions) |
| inthewild | 604 | [TrustAIRLab/in-the-wild-jailbreak-prompts](https://huggingface.co/datasets/TrustAIRLab/in-the-wild-jailbreak-prompts) jailbreaks; 3,647 regular prompts as benign |
| boundary | 456 | [3nesdeniz/agentic-prompt-injection-boundary-pairs](https://huggingface.co/datasets/3nesdeniz/agentic-prompt-injection-boundary-pairs), matched benign twins |
| jackhhao | 406 | [jackhhao/jailbreak-classification](https://huggingface.co/datasets/jackhhao/jailbreak-classification) train split; 481 benign |
| deepset | 188 | [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) train split; 327 benign |
| selfplay_r1..r5 | 1,339 | LLM-written attacks aimed at the current model's misses, relabelled by a judge; 1,079 benign look-alikes |
| rl_r1..r6 | 1,034 | confirmed misses from the GRPO attacker (`scripts/rl_attacker.py`); 888 benign twins (the same text with only the injected rule removed) |

Benign-only sources (label 0), chosen to look like the traffic a detector must not flag:

| source | rows | what it is |
|---|---|---|
| wildchat | 30,781 | [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) first user turns |
| dolly | 13,759 | [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) |
| oasst2 | 9,282 | [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) prompter turns |
| alpaca | 7,581 | [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) instruction + input, template stripped |
| orbench | 7,461 | [bench-llm/or-bench](https://huggingface.co/datasets/bench-llm/or-bench) 80k: safe prompts that sound unsafe |
| fineweb | 5,713 | [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) web paragraphs |
| emails | 5,550 | [Yale-LILY/aeslc](https://huggingface.co/datasets/Yale-LILY/aeslc) Enron emails |
| xtram | 5,398 | [xTRam1/safe-guard-prompt-injection](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection) benign rows only; its positives are too noisy |
| code | 4,734 | [pvduy/70k_evol_code_prompts](https://huggingface.co/datasets/pvduy/70k_evol_code_prompts) |
| wikipedia | 3,800 | [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) 2023-11 English paragraphs |
| marketing | 2,865 | [marketeam/marketing_user_prompts_unfiltered](https://huggingface.co/datasets/marketeam/marketing_user_prompts_unfiltered) |
| awesome | 2,248 | [fka/awesome-chatgpt-prompts](https://huggingface.co/datasets/fka/awesome-chatgpt-prompts): role-play system prompts that are not attacks |
| spml | 801 | [reshabhs/SPML_Chatbot_Prompt_Injection](https://huggingface.co/datasets/reshabhs/SPML_Chatbot_Prompt_Injection) benign rows |
| chatgpt_prompts | 339 | [MohamedRashad/ChatGPT-prompts](https://huggingface.co/datasets/MohamedRashad/ChatGPT-prompts) |

Rows shorter than 3 or longer than 20,000 characters are dropped. Licenses are listed per
dataset in `CITATIONS.md`; check them before commercial use.

## Labeling

A row is an injection when the text tries to instruct an AI model: override earlier
instructions, assume a persona to bypass rules, exfiltrate data, or trigger a tool call. Benign
rows include imperative but legitimate text (instruction-tuning prompts, system prompts,
refusal-bait questions) so the model learns that "do X" alone is not an attack.

## Known issues

- Source datasets disagree on what counts as an injection; jailbreak sets label a neighbouring
  task. Counts above are what survives filtering, not the upstream sizes.
- English only.
