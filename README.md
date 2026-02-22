# Career Assistance Chatbot (Domain-Specific LLM Fine-Tuning)

## Project Overview

This project implements a **domain-specific career assistance chatbot**
by fine-tuning a pre-trained Large Language Model (LLM) using **LoRA
(Low-Rank Adaptation)** and 4-bit quantization (QLoRA).

The chatbot specializes in: - Resume rewriting - ATS keyword
optimization - Mock interview preparation - Behavioral interview
coaching - General career advice

The fine-tuned model significantly outperforms the baseline model on
domain-specific tasks.

------------------------------------------------------------------------

## Model & Training Approach

-   Base Model: Pre-trained Hugging Face generative LLM
-   Fine-Tuning Method: LoRA using `peft`
-   Quantization: 4-bit (QLoRA)
-   Training Environment: Google Colab (Free GPU)
-   Dataset Size: 3,000 career-related Q&A pairs
-   Train/Validation Split: 2,700 / 300

------------------------------------------------------------------------

## Baseline Performance

  Metric       Value
  ------------ -------
  Loss         2.689
  Perplexity   14.72

------------------------------------------------------------------------

## Fine-Tuning Results

  Experiment   Epochs   Validation Loss   Perplexity
  ------------ -------- ----------------- ------------
  exp1         1        0.0435            1.04
  exp2         2        0.0438            1.04
  exp3         1        0.0440            1.04
  exp4         1        0.0467            1.05
  exp5         2        0.0477            1.05
  exp6         2        0.0475            1.05
  exp7         3        0.0498            1.05

### Key Findings

-   Fine-tuning reduced loss by \~98% compared to baseline.
-   Optimal performance achieved at **1 epoch**.
-   Training beyond 1 epoch showed signs of overfitting.

------------------------------------------------------------------------

## Installation & Setup

###  Clone Repository

``` bash
git clone https://github.com/rodwol/chatbot_careermate.git
```

### Install Dependencies

``` bash
pip install -r requirements.txt
```

### Run on Google Colab

-   Open the notebook in Colab
-   Enable GPU runtime
-   Run all cells sequentially

------------------------------------------------------------------------

## Training Pipeline

1.  Data preprocessing & formatting with special tokens:
    -   `<|system|>`
    -   `<|user|>`
    -   `<|assistant|>`
2.  Tokenization using Hugging Face tokenizer
3.  LoRA configuration using `peft`
4.  Fine-tuning with Trainer API
5.  Evaluation (Loss & Perplexity)
6.  Model inference testing

------------------------------------------------------------------------

## User Interface

The model is deployed using **Gradio** for interactive testing.

Features: - Clean text input - Real-time response generation -
Career-focused assistant persona


## Demo Video

https://www.canva.com/design/DAHCFaX5McQ/oFIMC7DquW4KWp6PQtMqGg/edit?utm_content=DAHCFaX5McQ&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

------------------------------------------------------------------------

## Colab Notebook
https://lightning.ai/rgoniche/deploy-model-project/studios/superb-apricot-mhr4/code?source=copylink

------------------------------------------------------------------------

------------------------------------------------------------------------

## Hugging Face Access
https://huggingface.co/spaces/helinow/Careermate

------------------------------------------------------------------------

## Future Improvements

-   Add BLEU/ROUGE evaluation metrics
-   Expand dataset size
-   Add learning rate comparison experiments
-   Deploy as web-hosted API

------------------------------------------------------------------------

## Author

Rodas Goniche
