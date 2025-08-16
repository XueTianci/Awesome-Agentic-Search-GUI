# Awesome Agentic Search and GUI Agents

This repository is a curated collection of research papers and resources on building next-generation GUI Agents and Agentic Search agents.

GUI Agents – Autonomous systems that can interact with graphical user interfaces by perceiving UI elements (e.g., buttons, inputs, menus) and performing human-like actions such as clicking, typing, and dragging to accomplish tasks.

Agentic Search / Deep Research – AI agents that autonomously plan, reason, and synthesize information across sources for complex tasks.


# Awesome Papers

## Table of Contents
  
1. [Agentic Search](#agentic-search)  
2. [GUI Agent](#gui-agent)


## Agentic Search

| Legend | ✔️ Implemented | — Not present |
|--------|:--------------:|:-------------:|

| Title | SFT | RL | Base Model | Environment | Evaluation | reward |
|---|:-:|:-:|---|---|---|---|
| WebDancer | ✔️ | GRPO | QwQ-32B |  | GAIA | Rule-based |


## Computer-Use Agent

| Legend | ✔️ Implemented | — Not present |
|--------|:--------------:|:-------------:|

| Title | Date | SFT | RL | Base Model | Environment | Evaluation | reward |
|------------------------------|---|:-:|:-:|---|---|---|---|
| [SEAgent: Self-Evolving Computer Use Agent with Autonomous Learning from Experience](https://www.arxiv.org/pdf/2508.04700) [![GitHub Stars](https://img.shields.io/github/stars/SunzeY/SEAgent?style=social)](https://github.com/SunzeY/SEAgent)| Aug, 2025 | ✔️ | ✔️ GRPO | UI-TARS-7B-DPO | Computer | OSWorld | model-based |
| [OpenCUA: Open Foundations for Computer-Use Agents](https://arxiv.org/abs/2508.09123) [![GitHub Stars](https://img.shields.io/github/stars/xlang-ai/OpenCUA?style=social)](https://github.com/xlang-ai/OpenCUA)| Aug, 2025 | ✔️ | — |  Qwen2.5-VL-7B-Instruction | Computer | OSWorld-Verified, WindowsAgentArena | — | — |
| [WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning](https://arxiv.org/abs/2411.02337) [![GitHub Stars](https://img.shields.io/github/stars/THUDM/WebRL?style=social)](https://github.com/THUDM/WebRL) | Nov, 2024 | — | ✔️ Curriculum-RL | Llama-3.1-8B, GLM-4-9B | Web | WebArena-Lite | model-based |
| [WebAgent-R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning](https://arxiv.org/abs/2505.16421) [![GitHub Stars](https://img.shields.io/github/stars/weizhepei/WebAgent-R1?style=social)](https://github.com/weizhepei/WebAgent-R1) | May, 2025 | — | ✔️ GRPO | Qwen-2.5-3B, Llama-3.1-8B | Web | WebArena-Lite | rule-based |
| [ARPO: End-to-End Policy Optimization for GUI Agents with Experience Replay](https://arxiv.org/abs/2505.16282) [![GitHub Stars](https://img.shields.io/github/stars/dvlab-research/ARPO?style=social)](https://github.com/dvlab-research/ARPO) | May 2025 | — | ✔️ ARPO | UI-TARS-1.5-7B | Computer | OSWorld | rule-based |
| [How to Train Your LLM Web Agent: A Statistical Diagnosis](https://arxiv.org/abs/2507.04103) | Jul 2025 | ✔️ SFT | ✔️ GRPO | Llama-3.1-8B | Web | WorkArena, MiniWoB++ | rule-based |
| [ZeroGUI: Automating Online GUI Learning at Zero Human Cost](https://arxiv.org/abs/2505.23762) [![GitHub Stars](https://img.shields.io/github/stars/OpenGVLab/ZeroGUI?style=social)](https://github.com/OpenGVLab/ZeroGUI) | May 2025 | — | ✔️ GRPO | UI-TARS-7B-DPO, Aguvi | Computer, Android | OSWorld, AndroidLab | model-based |
| [Efficient Agent Training for Computer Use](https://arxiv.org/abs/2505.13909) | May 2025 | ✔️ | — | Qwen2.5-VL-72B-Instruct | Computer (windows) | WindowsAgentArena-V2 | — |
| [MobileGUI-RL: Advancing Mobile GUI Agent through Reinforcement Learning in Online Environment](https://arxiv.org/abs/2507.05720) | Jul 2025 | — | ✔️ GRPO | Qwen2.5-VL-7B-Instruct, Qwen2.5-VL-32B-Instruct | Android | AndroidWorld, Android-in-theWild | rule-based |
