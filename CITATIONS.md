# Training data sources

Every dataset `scripts/build_dataset.py` pulls from, with the license from the Hugging Face API
on 2026-10-24. Two sources are non-commercial and five state no license at all; both matter if
you redistribute the training set (the model weights themselves are MIT). Per-source row counts
and how each is filtered are in `dataset_card.md`.

| Dataset | License | Used as |
|---|---|---|
| [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | Apache-2.0 | both labels |
| [S-Labs/prompt-injection-dataset](https://huggingface.co/datasets/S-Labs/prompt-injection-dataset) | MIT | both labels |
| [neuralchemy/Prompt-injection-dataset](https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset) | Apache-2.0 (sources: HackAPrompt CC BY 4.0, HarmBench MIT) | both labels |
| [jackhhao/jailbreak-classification](https://huggingface.co/datasets/jackhhao/jailbreak-classification) | Apache-2.0 | both labels |
| [3nesdeniz/agentic-prompt-injection-5k](https://huggingface.co/datasets/3nesdeniz/agentic-prompt-injection-5k) | CC BY 4.0 | both labels |
| [3nesdeniz/agentic-prompt-injection-boundary-pairs](https://huggingface.co/datasets/3nesdeniz/agentic-prompt-injection-boundary-pairs) | CC BY 4.0 | both labels |
| [TrustAIRLab/in-the-wild-jailbreak-prompts](https://huggingface.co/datasets/TrustAIRLab/in-the-wild-jailbreak-prompts) | MIT | both labels |
| [microsoft/llmail-inject-challenge](https://huggingface.co/datasets/microsoft/llmail-inject-challenge) | MIT | injections |
| [Lakera/gandalf_ignore_instructions](https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions) | MIT | injections |
| [imoxto/prompt_injection_hackaprompt_gpt35](https://huggingface.co/datasets/imoxto/prompt_injection_hackaprompt_gpt35) | none stated (upstream HackAPrompt is CC BY 4.0) | injections |
| [xTRam1/safe-guard-prompt-injection](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection) | none stated | benign only |
| [reshabhs/SPML_Chatbot_Prompt_Injection](https://huggingface.co/datasets/reshabhs/SPML_Chatbot_Prompt_Injection) | MIT | benign only |
| [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) | ODC-BY, AI2 ImpACT terms | benign |
| [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | CC BY-SA 3.0 | benign |
| [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) | Apache-2.0 | benign |
| [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) | CC BY-NC 4.0 (non-commercial) | benign |
| [bench-llm/or-bench](https://huggingface.co/datasets/bench-llm/or-bench) | CC BY 4.0 | benign |
| [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | ODC-BY | benign |
| [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) | CC BY-SA 3.0 | benign |
| [Yale-LILY/aeslc](https://huggingface.co/datasets/Yale-LILY/aeslc) | none stated ("unknown" on the card) | benign |
| [pvduy/70k_evol_code_prompts](https://huggingface.co/datasets/pvduy/70k_evol_code_prompts) | none stated | benign |
| [marketeam/marketing_user_prompts_unfiltered](https://huggingface.co/datasets/marketeam/marketing_user_prompts_unfiltered) | Apache-2.0 | benign |
| [fka/awesome-chatgpt-prompts](https://huggingface.co/datasets/fka/awesome-chatgpt-prompts) | CC0-1.0 | benign |
| [MohamedRashad/ChatGPT-prompts](https://huggingface.co/datasets/MohamedRashad/ChatGPT-prompts) | none stated | benign |

Evaluation only (never trained on):

| Dataset | License |
|---|---|
| [Athithya117/prompt-injection-eval-suite](https://huggingface.co/datasets/Athithya117/prompt-injection-eval-suite) | mixed, per config (see its card) |
| [lmsys/toxic-chat](https://huggingface.co/datasets/lmsys/toxic-chat) | CC BY-NC 4.0 (non-commercial) |
