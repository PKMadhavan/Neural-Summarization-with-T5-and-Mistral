# Summarization Experiments with T5 and Mistral

This project explores abstractive text summarization with encoder‑decoder transformers and instruction‑tuned large language models, combining supervised fine‑tuning and prompt‑based generation on news and long‑form essay datasets. 

## Features

- Runs off‑the‑shelf and fine‑tuned `t5-small` on a CNN/DailyMail‑style news subset, comparing summary quality and fluency. 
- Fine‑tunes `t5-small` on an essay collection (train/validation/test split) and evaluates on held‑out essays. 
- Uses a 4‑bit quantized `mistralai/Mistral-7B-Instruct-v0.2` model for prompt‑only summarization of long essays.
- Evaluates models with ROUGE‑1/2, BERTScore‑F1, and GPT‑2 perplexity, plus qualitative side‑by‑side comparisons of generated summaries versus references.

## Notebooks

- `summarization_t5_cnn_dailymail.ipynb`  
  - Loads a sampled news dataset, tokenizes articles and highlights, and runs both base and fine‑tuned `t5-small` to generate summaries.
  - Computes ROUGE, BERTScore, and perplexity, and prints example articles with reference and model summaries. 

- `summarization_t5_aeon_essays.ipynb`  
  - Loads an essays CSV, builds a train/validation/test split, and fine‑tunes `t5-small` with `Seq2SeqTrainer`.
  - Evaluates off‑the‑shelf vs fine‑tuned models on the essay test split and reports metric differences and qualitative examples. 

- `summarization_mistral_llm.ipynb`  
  - Loads a 4‑bit Mistral‑7B model with bitsandbytes, defines two summarization prompts (plain summary and academic‑style abstract), and generates summaries for long essays.
  - Compares LLM outputs qualitatively to T5 and reference summaries.

## How to Run

1. Install dependencies: `transformers`, `datasets`, `evaluate`, `rouge-score`, `pandas`, `accelerate`, `bitsandbytes`, `bertscore`, `py7zr`, and `torch`. 
2. Provide the required CSV files (news subset and essays dataset) in the paths expected by each notebook, or adjust the load paths.
3. Open each notebook in Jupyter or Colab and run all cells to reproduce training, evaluation, and sample generations.

## Key Takeaways

- Fine‑tuning T5 improves semantic alignment (higher BERTScore and lower perplexity) even when ROUGE gains are modest or mixed. 
- Domain‑specific fine‑tuning is important when moving from news to essay‑style content; off‑the‑shelf models often underperform or fail.
- Instruction‑tuned LLMs like Mistral‑7B can produce fluent, controllable summaries via prompting alone, but may omit fine‑grained details compared to supervised T5 models. [file:24]
