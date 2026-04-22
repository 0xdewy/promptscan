# Training Data

PromptScan is trained on the following Hugging Face datasets. Check each dataset's license before commercial use.

| Dataset | License | Notes |
|---------|---------|-------|
| [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | Apache 2.0 | |
| [S-Labs/prompt-injection-dataset](https://huggingface.co/datasets/S-Labs/prompt-injection-dataset) | MIT | |
| [neuralchemy/Prompt-injection-dataset](https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset) | Apache 2.0 | Sources: HackAPrompt (CC BY 4.0), HarmBench (MIT) |
| [Octavio-Santana/prompt-injection-attack-detection-multilingual](https://huggingface.co/datasets/Octavio-Santana/prompt-injection-attack-detection-multilingual) | GPL | |
| [imoxto/prompt_injection_hackaprompt_gpt35](https://huggingface.co/datasets/imoxto/prompt_injection_hackaprompt_gpt35) | — | |
| [imoxto/prompt_injection_cleaned_dataset-v2](https://huggingface.co/datasets/imoxto/prompt_injection_cleaned_dataset-v2) | — | |
| [cgoosen/prompt_injection_ctf_dataset_2](https://huggingface.co/datasets/cgoosen/prompt_injection_ctf_dataset_2) | — | |
| [MohamedRashad/ChatGPT-prompts](https://huggingface.co/datasets/MohamedRashad/ChatGPT-prompts) | — | Safe prompts only (`is_injection=False`) |
| [pvduy/70k_evol_code_prompts](https://huggingface.co/datasets/pvduy/70k_evol_code_prompts) | — | Safe coding prompts (`is_injection=False`) |
| [marketeam/marketing_user_prompts_unfiltered](https://huggingface.co/datasets/marketeam/marketing_user_prompts_unfiltered) | — | Safe marketing prompts (`is_injection=False`) |
| [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) | Apache 2.0 | Safe diverse instruction-following prompts (`is_injection=False`) |

Mirrored datasets (same content, different repo): `cyberec/Prompt-injection-dataset`, `cyberec/Prompt-injection-dataset2`, `adfksfasbjsdk/Prompt-injection-dataset` all mirror `neuralchemy/Prompt-injection-dataset`.
