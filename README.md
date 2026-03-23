# 🧠 Multi-Agent Math Reasoning (Dual-Track Architecture)
A multi-agent AI pipeline using 14B/7B LLMs and the Z3 Theorem Prover to autonomously solve, audit, and self-correct complex Olympiad math problems.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/Paper-NeurIPS_Draft-red.svg)](#citation)

This repository contains the official implementation of the **Dual-Track Architecture**, a neuro-symbolic multi-agent framework designed to solve complex mathematical Olympiad (AIMO) problems. 

By treating Large Language Models (LLMs) as modular components within a strictly regulated computational graph, this pipeline mitigates common LLM failure modes such as "Cognitive Anchoring," mode collapse, and syntax hallucinations during long-horizon reasoning tasks.

## 🌟 Core Innovations

* **🕵️ Zero-Shot Domain Scouting (The "Recon Muzzle"):** A rigidly constrained 7B agent pre-conditions the 14B primary solver by extracting domain parameters and mathematical traps, shielding the solver from conversation-filler and hallucinated constraints.
* **⚖️ Process Advantage Verifier (PAV):** A dynamic mid-trajectory Shannon entropy metric that detects when the model is statistically "confused," aggressively pruning doomed reasoning branches to save GPU compute.
* **🚨 The Crisis Diagnostician:** A dynamic state-intervention protocol. If the primary solver hits a frustration threshold (e.g., continuous `SyntaxErrors`), a 7B Diagnostician performs a technical post-mortem and injects a 1-sentence tactical command to force an algorithmic pivot.
* **🔒 Agentic Z3 Verification:** Bridges the gap between neural generation and symbolic formal logic. The pipeline prompts the model to write formal satisfiability proofs in `z3-solver` to mathematically verify outputs and override blind majority-vote consensus.

## 📂 Repository Structure

The architecture operates on a **Dual-Track** system to manage I/O bottlenecks in highly parallel Jupyter/Kaggle environments:

* `lvties4.py` **(Debug Track):** Designed for ablation and development. Features verbose telemetry, tracks Lines of Code (LOC), byte size, and prints internal Crisis Diagnostician reports.
* `lvties5.py` **(Submission Track):** A stateless, highly-parallel production track optimized for competitive environments. Disables history managers to prevent SQLite locks across 16+ parallel workers.
* `test.py` **(Benchmark Suite):** Contains the 50-problem test suite (including known "Traps" like the Coprime Totient and Derangement recursion limits) for local evaluation.

## ⚙️ Hardware & Environment Requirements

This pipeline is aggressively optimized for **H100 / A100 GPU** environments and relies on the `vLLM` inference engine for maximum throughput.

**Required Models:**
* **Primary Solver:** `maa-deepseek-qwen-14b-ties-merged`
* **Critic/Diagnostician:** `maa-deepseek-qwen-7b-ties-merged` (or equivalent `deepseek-r1-distill-qwen-7b`)

## 🚀 Quick Start

**1. Clone the repository:**
```bash
git clone [https://github.com/mavergonzado/Multi-Agent-Math-Reasoning.git](https://github.com/mavergonzado/Multi-Agent-Math-Reasoning.git)
cd Multi-Agent-Math-Reasoning
