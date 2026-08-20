# Implicit Cultural Alignment Reward Model

This repository provides the core architecture and evaluation pipeline for the Implicit Cultural Alignment Reward Model. Built upon the Phi-3.5-vision architecture and optimized with DeepSpeed and LoRA, this framework is designed to evaluate and align Multimodal Large Language Models (MLLMs) with nuanced cultural expectations.

## Architecture

![Architecture Overview](./assets/architecture.png)

Our framework integrates an Implicit Cultural Probe with a Skip-connection Cross-Attention (SkipCA) mechanism[cite: 1]. This design enables late-stage semantic features to directly attend to early-stage visual representations, better preserving culturally salient details[cite: 1]. By bypassing autoregressive text generation, the model processes each evaluation efficiently in 0.21 seconds under our local inference setup, achieving a 10x speedup over standard VQA-based evaluators[cite: 1].

## Dataset

This project utilizes the **CulturalFrames** dataset to evaluate cultural biases and human expectations in text-to-image generation and multimodal understanding[cite: 1].

* **Dataset Link**: [CulturalFrames on Hugging Face](https://huggingface.co/datasets/mair-lab/CulturalFrames)

## Model Weights

To maintain a lightweight and clean Git history, the fine-tuned LoRA adapters and model checkpoints are hosted on Hugging Face.

* **Model Repository**: [Bensonch/phi35_cultural_reward](https://huggingface.co/Bensonch/phi35_cultural_reward)

### Downloading the Weights

You can easily pull the required weights directly into your local `checkpoints` directory using the Hugging Face CLI:

```bash
hf download Bensonch/phi35_cultural_reward --local-dir ./checkpoints
