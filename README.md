# Awesome Agentic Search and GUI Agents

This repository is a curated collection of research papers and resources on building next-generation GUI Agents and Agentic Search agents.

GUI Agents – Autonomous systems that can interact with graphical user interfaces by perceiving UI elements (e.g., buttons, inputs, menus) and performing human-like actions such as clicking, typing, and dragging to accomplish tasks.

Agentic Search / Deep Research – AI agents that autonomously plan, reason, and synthesize information across sources for complex tasks.


# Awesome Papers

## Table of Contents
  
1. [Agentic Search](#agentic-search)  
2. [GUI Agent](#gui-agent)


## Agentic Search

<details>
<summary>
  <a href="https://arxiv.org/abs/2508.07976">Beyond Ten Turns: Unlocking Long-Horizon Agentic Search with Large-Scale Asynchronous RL</a>
  <a href="https://github.com/inclusionAI/ASearcher"><img src="https://img.shields.io/github/stars/inclusionAI/ASearcher?style=social" alt="GitHub Stars"/></a>
</summary>

- Date: Aug, 2025  
- Env: API, Browser  
- RL: GRPO  
- Base Model: Qwen2.5-7B, Qwen2.5-14B 
- Benchmark: Single-Hop QA, Multi-Hop QA, GAIA, xBench-DeepSearch, Frames
- reward: Model-based  

**TLDR**:  
Recent advancements in LLM-based agents have demonstrated remarkable capabilities in handling complex, knowledge-intensive tasks by integrating external tools. Among diverse choices of tools, search tools play a pivotal role in accessing vast external knowledge. However, open-source agents still fall short of achieving expert-level Search Intelligence, the ability to resolve ambiguous queries, generate precise searches, analyze results, and conduct thorough exploration. Existing approaches fall short in scalability, efficiency, and data quality. For example, small turn limits in existing online RL methods, e.g. <=10, restrict complex strategy learning. This paper introduces ASearcher, an open-source project for large-scale RL training of search agents. Our key contributions include: (1) Scalable fully asynchronous RL training that enables long-horizon search while maintaining high training efficiency. (2) A prompt-based LLM agent that autonomously synthesizes high-quality and challenging QAs, creating a large-scale QA dataset. Through RL training, our prompt-based QwQ-32B agent achieves substantial improvements, with 46.7% and 20.8% Avg@4 gains on xBench and GAIA, respectively. Notably, our agent exhibits extreme long-horizon search, with tool calls exceeding 40 turns and output tokens exceeding 150k during training time. With a simple agent design and no external LLMs, ASearcher-Web-QwQ achieves Avg@4 scores of 42.1 on xBench and 52.8 on GAIA, surpassing existing open-source 32B agents. We open-source our models, training data, and codes in this https URL. 

</details>


## GUI Agent


<details>
<summary>
  <a href="https://arxiv.org/abs/2508.04482">OS Agents: A Survey on MLLM-based Agents for General Computing Devices Use</a>
</summary>

- Date: Aug, 2025  

**TLDR**:  
Vision-language models have demonstrated impressive capabilities as computer-use agents (CUAs) capable of automating diverse computer tasks. As their commercial potential grows, critical details of the most capable CUA systems remain closed. As these agents will increasingly mediate digital interactions and execute consequential decisions on our behalf, the research community needs access to open CUA frameworks to study their capabilities, limitations, and risks. To bridge this gap, we propose OpenCUA, a comprehensive open-source framework for scaling CUA data and foundation models. Our framework consists of: (1) an annotation infrastructure that seamlessly captures human computer-use demonstrations; (2) AgentNet, the first large-scale computer-use task dataset spanning 3 operating systems and 200+ applications and websites; (3) a scalable pipeline that transforms demonstrations into state-action pairs with reflective long Chain-of-Thought reasoning that sustain robust performance gains as data scales. Our end-to-end agent models demonstrate strong performance across CUA benchmarks. In particular, OpenCUA-32B achieves an average success rate of 34.8% on OSWorld-Verified, establishing a new state-of-the-art (SOTA) among open-source models and surpassing OpenAI CUA (GPT-4o). Further analysis confirms that our approach generalizes well across domains and benefits significantly from increased test-time computation. We release our annotation tool, datasets, code, and models to build open foundations for further CUA research. 

</details>

<details>
<summary>
  <a href="https://arxiv.org/abs/2508.03923">CoAct-1: Computer-using Agents with Coding as Actions</a>
</summary>

- Date: Aug, 2025  
- Env: Computer  
- Base Model: o3, o4-mini, computer-use-preview
- Benchmark: OSWorld

**TLDR**:  
Autonomous agents that operate computers via Graphical User Interfaces (GUIs) often struggle with efficiency and reliability on complex, long-horizon tasks. While augmenting these agents with planners can improve task decomposition, they remain constrained by the inherent limitations of performing all actions through GUI manipulation, leading to brittleness and inefficiency. In this work, we introduce a more robust and flexible paradigm: enabling agents to use coding as a enhanced action. We present CoAct-1, a novel multi-agent system that synergistically combines GUI-based control with direct programmatic execution. CoAct-1 features an Orchestrator that dynamically delegates subtasks to either a conventional GUI Operator or a specialized Programmer agent, which can write and execute Python or Bash scripts. This hybrid approach allows the agent to bypass inefficient GUI action sequences for tasks like file management and data processing, while still leveraging visual interaction when necessary. We evaluate our system on the challenging OSWorld benchmark, where CoAct-1 achieves a new state-of-the-art success rate of 60.76%, significantly outperforming prior methods. Furthermore, our approach dramatically improves efficiency, reducing the average number of steps required to complete a task to just 10.15, compared to 15 for leading GUI agents. Our results demonstrate that integrating coding as a core action provides a more powerful, efficient, and scalable path toward generalized computer automation.

</details>


<details>
<summary>
  <a href="https://arxiv.org/abs/2508.03700">MagicGUI: A Foundational Mobile GUI Agent with Scalable Data Pipeline and Reinforcement Fine-tuning</a>
</summary>

- Date: Aug, 2025  
- Env: Mobile 
- Base Model: Qwen‑VL  
- Benchmark: GUI-Odyssey, AndroidControl, Magic-RICH

**TLDR**:  
This paper presents MagicGUI, a foundational mobile GUI agent designed to address critical challenges in perception, grounding, and reasoning within real-world mobile GUI environments. The framework is underpinned by following six key components: (1) a comprehensive and accurate dataset, constructed via the scalable GUI Data Pipeline, which aggregates the largest and most diverse GUI-centric multimodal data to date from open-source repositories, automated crawling, and targeted manual annotation; (2) enhanced perception and grounding capabilities, facilitating fine-grained multimodal alignment for UI element referencing, grounding, and screen comprehension; (3) a comprehensive and unified action space, encompassing both fundamental UI operations and complex interactive intents to support human-agent interactions; (4) planning-oriented reasoning mechanisms that enable the model to decompose complex user instructions into sequential actions with explicit intermediate meta-paln reasoning; (5) an iterative two-stage training procedure, combining large-scale continue pre-training on 7.8M samples with reinforcement fine-tuning utilizing a spatially enhanced composite reward and dual filtering strategy; and (6) competitive performance on both the proprietary Magic-RICH benchmark and over a dozen public benchmarks, achieving superior performance across GUI perception and agent tasks, while demonstrating robust generalization and real-world deployment potential in practical mobile GUI scenarios, as detailed in Figure 1.

</details>


</details>


<details>
<summary>
  <a href="https://arxiv.org/abs/2508.09123">OpenCUA: Open Foundations for Computer-Use Agents</a>
  <a href="https://github.com/xlang-ai/OpenCUA"><img src="https://img.shields.io/github/stars/xlang-ai/OpenCUA?style=social" alt="GitHub Stars"/></a>
</summary>

- Date: Aug, 2025  
- Env: Computer  
- Base Model: Qwen2.5-VL-7B-Instruction  
- Benchmark: OSWorld-Verified, WindowsAgentArena  

**TLDR**:  
Vision-language models have demonstrated impressive capabilities as computer-use agents (CUAs) capable of automating diverse computer tasks. As their commercial potential grows, critical details of the most capable CUA systems remain closed. As these agents will increasingly mediate digital interactions and execute consequential decisions on our behalf, the research community needs access to open CUA frameworks to study their capabilities, limitations, and risks. To bridge this gap, we propose OpenCUA, a comprehensive open-source framework for scaling CUA data and foundation models. Our framework consists of: (1) an annotation infrastructure that seamlessly captures human computer-use demonstrations; (2) AgentNet, the first large-scale computer-use task dataset spanning 3 operating systems and 200+ applications and websites; (3) a scalable pipeline that transforms demonstrations into state-action pairs with reflective long Chain-of-Thought reasoning that sustain robust performance gains as data scales. Our end-to-end agent models demonstrate strong performance across CUA benchmarks. In particular, OpenCUA-32B achieves an average success rate of 34.8% on OSWorld-Verified, establishing a new state-of-the-art (SOTA) among open-source models and surpassing OpenAI CUA (GPT-4o). Further analysis confirms that our approach generalizes well across domains and benefits significantly from increased test-time computation. We release our annotation tool, datasets, code, and models to build open foundations for further CUA research. 

</details>


<details>
<summary>
  <a href="https://www.arxiv.org/abs/2508.04700">SEAgent: Self-Evolving Computer Use Agent with Autonomous Learning from Experience</a>
  <a href="https://github.com/SunzeY/SEAgent"><img src="https://img.shields.io/github/stars/SunzeY/SEAgent?style=social" alt="GitHub Stars"/></a>
</summary>

- Date: Aug, 2025  
- Env: Computer  
- RL: GRPO  
- Base Model: UI-TARS-7B-DPO  
- Benchmark: OSWorld  
- reward: Model-based  

**TLDR**:  
Repurposing large vision-language models (LVLMs) as computer use agents (CUAs) has led to substantial breakthroughs, primarily driven by human-labeled data. However, these models often struggle with novel and specialized software, particularly in scenarios lacking human annotations. To address this challenge, we propose SEAgent, an agentic self-evolving framework enabling CUAs to autonomously evolve through interactions with unfamiliar software. Specifically, SEAgent empowers computer-use agents to autonomously master novel software environments via experiential learning, where agents explore new software, learn through iterative trial-and-error, and progressively tackle auto-generated tasks organized from simple to complex. To achieve this goal, we design a World State Model for step-wise trajectory assessment, along with a Curriculum Generator that generates increasingly diverse and challenging tasks. The agent's policy is updated through experiential learning, comprised of adversarial imitation of failure actions and Group Relative Policy Optimization (GRPO) on successful ones. Furthermore, we introduce a specialist-to-generalist training strategy that integrates individual experiential insights from specialist agents, facilitating the development of a stronger generalist CUA capable of continuous autonomous evolution. This unified agent ultimately achieves performance surpassing ensembles of individual specialist agents on their specialized software. We validate the effectiveness of SEAgent across five novel software environments within OS-World. Our approach achieves a significant improvement of 23.2% in success rate, from 11.3% to 34.5%, over a competitive open-source CUA, i.e., UI-TARS.  

</details>


<details>
<summary>
  <a href="https://arxiv.org/abs/2507.05720">MobileGUI-RL: Advancing Mobile GUI Agent through Reinforcement Learning in Online Environment</a>
</summary>

- Date: Jul, 2025  
- Env: Android  
- Method: GRPO  
- Base Model: Qwen2.5-VL-7B-Instruct, Qwen2.5-VL-32B-Instruct  
- Benchmark: AndroidWorld, Android-in-theWild  
- Paradigm: Rule-based  

**TLDR**:  
Recently, there has been a surge of vision-based GUI agents designed to automate everyday mobile and web tasks. These agents interpret raw GUI screenshots and autonomously decide where to click, scroll, or type, which bypasses handcrafted rules and app-specific APIs. However, most existing methods trained GUI agent in the offline environment using pre-collected trajectories. This approach limits scalability, causes overfitting to specific UI templates, and leads to brittle policies when faced with unseen environment. We present MobileGUI-RL, a scalable framework that trains GUI agent in online environment. MobileGUI-RL contains two key components. It (i) synthesizes a curriculum of learnable tasks through self-exploration and filtering, and (ii) adapts GRPO to GUI navigation with trajectory-aware advantages and composite rewards that balance task success and execution efficiency. Experiments on three online mobile-agent benchmarks show consistent gains, validating the effectiveness of our approach.  

</details>


<details>
<summary>
  <a href="https://arxiv.org/abs/2507.04103">How to Train Your LLM Web Agent: A Statistical Diagnosis</a>
</summary>

- Date: Jul, 2025  
- Env: Web  
- RL: GRPO  
- Base Model: Llama-3.1-8B  
- Benchmark: WorkArena, MiniWoB++  
- Reward: Rule-based  

**TLDR**:  
LLM-based web agents have recently made significant progress, but much of it has occurred in closed-source systems, widening the gap with open-source alternatives. Progress has been held back by two key challenges: first, a narrow focus on single-step tasks that overlooks the complexity of multi-step web interactions; and second, the high compute costs required to post-train LLM-based web agents. To address this, we present the first statistically grounded study on compute allocation for LLM web-agent post-training. Our approach uses a two-stage pipeline, training a Llama 3.1 8B student to imitate a Llama 3.3 70B teacher via supervised fine-tuning (SFT), followed by on-policy reinforcement learning. We find this process highly sensitive to hyperparameter choices, making exhaustive sweeps impractical. To spare others from expensive trial-and-error, we sample 1,370 configurations and use bootstrapping to estimate effective hyperparameters. Our results show that combining SFT with on-policy RL consistently outperforms either approach alone on both WorkArena and MiniWob++. Further, this strategy requires only 55% of the compute to match the peak performance of pure SFT on MiniWob++, effectively pushing the compute-performance Pareto frontier, and is the only strategy that can close the gap with closed-source models. 

</details>


<details>
<summary>
  <a href="https://arxiv.org/abs/2505.16282">ARPO: End-to-End Policy Optimization for GUI Agents with Experience Replay</a>
  <a href="https://github.com/dvlab-research/ARPO"><img src="https://img.shields.io/github/stars/dvlab-research/ARPO?style=social" alt="GitHub Stars"/></a>
</summary>

- Date: May, 2025  
- Env: Computer  
- RL: ARPO  
- Base Model: UI-TARS-1.5-7B  
- Benchmark: OSWorld  
- Reward: Rule-based  

**TLDR**:  
Training large language models (LLMs) as interactive agents for controlling graphical user interfaces (GUIs) presents a unique challenge to optimize long-horizon action sequences with multimodal feedback from complex environments. While recent works have advanced multi-turn reinforcement learning (RL) for reasoning and tool-using capabilities in LLMs, their application to GUI-based agents remains relatively underexplored due to the difficulty of sparse rewards, delayed feedback, and high rollout costs. In this paper, we investigate end-to-end policy optimization for vision-language-based GUI agents with the aim of improving performance on complex, long-horizon computer tasks. We propose Agentic Replay Policy Optimization (ARPO), an end-to-end RL approach that augments Group Relative Policy Optimization (GRPO) with a replay buffer to reuse the successful experience across training iterations. To further stabilize the training process, we propose a task selection strategy that filters tasks based on baseline agent performance, allowing the agent to focus on learning from informative interactions. Additionally, we compare ARPO with offline preference optimization approaches, highlighting the advantages of policy-based methods in GUI environments. Experiments on the OSWorld benchmark demonstrate that ARPO achieves competitive results, establishing a new performance baseline for LLM-based GUI agents trained via reinforcement learning. Our findings underscore the effectiveness of reinforcement learning for training multi-turn, vision-language GUI agents capable of managing complex real-world UI interactions.

</details>


<details>
<summary>
  <a href="https://arxiv.org/abs/2505.16421">WebAgent-R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning</a>
  <a href="https://github.com/weizhepei/WebAgent-R1"><img src="https://img.shields.io/github/stars/weizhepei/WebAgent-R1?style=social" alt="GitHub Stars"/></a>
</summary>

- Date: May, 2025  
- Env: Web  
- RL: GRPO  
- Base Model: Qwen-2.5-3B, Llama-3.1-8B  
- Benchmark: WebArena-Lite  
- Reward: Rule-based  

**TLDR**:  
While reinforcement learning (RL) has demonstrated remarkable success in enhancing large language models (LLMs), it has primarily focused on single-turn tasks such as solving math problems. Training effective web agents for multi-turn interactions remains challenging due to the complexity of long-horizon decision-making across dynamic web interfaces. In this work, we present WebAgent-R1, a simple yet effective end-to-end multi-turn RL framework for training web agents. It learns directly from online interactions with web environments by asynchronously generating diverse trajectories, entirely guided by binary rewards depending on task success. Experiments on the WebArena-Lite benchmark demonstrate the effectiveness of WebAgent-R1, boosting the task success rate of Qwen-2.5-3B from 6.1% to 33.9% and Llama-3.1-8B from 8.5% to 44.8%, significantly outperforming existing state-of-the-art methods and strong proprietary models such as OpenAI o3. In-depth analyses reveal the effectiveness of the thinking-based prompting strategy and test-time scaling through increased interactions for web tasks. We further investigate different RL initialization policies by introducing two variants, namely WebAgent-R1-Zero and WebAgent-R1-CoT, which highlight the importance of the warm-up training stage (i.e., behavior cloning) and provide insights on incorporating long chain-of-thought (CoT) reasoning in web agents. 

</details>


<details>
<summary>
  <a href="https://arxiv.org/abs/2505.23762">ZeroGUI: Automating Online GUI Learning at Zero Human Cost</a>
  <a href="https://github.com/OpenGVLab/ZeroGUI"><img src="https://img.shields.io/github/stars/OpenGVLab/ZeroGUI?style=social" alt="GitHub Stars"/></a>
</summary>

- Date: May, 2025  
- Env: Computer, Android  
- RL: GRPO  
- Base Model: UI-TARS-7B-DPO, Aguvi  
- Benchmark: OSWorld, AndroidLab  
- Reward: Model-based  

**TLDR**:  
The rapid advancement of large Vision-Language Models (VLMs) has propelled the development of pure-vision-based GUI Agents, capable of perceiving and operating Graphical User Interfaces (GUI) to autonomously fulfill user instructions. However, existing approaches usually adopt an offline learning framework, which faces two core limitations: (1) heavy reliance on high-quality manual annotations for element grounding and action supervision, and (2) limited adaptability to dynamic and interactive environments. To address these limitations, we propose ZeroGUI, a scalable, online learning framework for automating GUI Agent training at Zero human cost. Specifically, ZeroGUI integrates (i) VLM-based automatic task generation to produce diverse training goals from the current environment state, (ii) VLM-based automatic reward estimation to assess task success without hand-crafted evaluation functions, and (iii) two-stage online reinforcement learning to continuously interact with and learn from GUI environments. Experiments on two advanced GUI Agents (UI-TARS and Aguvis) demonstrate that ZeroGUI significantly boosts performance across OSWorld and AndroidLab environments.

</details>


<details>
<summary>
  <a href="https://arxiv.org/abs/2505.13909">Efficient Agent Training for Computer Use</a>
  <a href="https://github.com/GAIR-NLP/PC-Agent-E"><img src="https://img.shields.io/github/stars/GAIR-NLP/PC-Agent-E?style=social" alt="GitHub Stars"/></a>
</summary>

- Date: May, 2025  
- Env: Computer (Windows)  
- Base Model: Qwen2.5-VL-72B-Instruct  
- Benchmark: WindowsAgentArena-V2  

**TLDR**:  
Scaling up high-quality trajectory data has long been a critical bottleneck for developing human-like computer use agents. We introduce PC Agent-E, an efficient agent training framework that significantly reduces reliance on large-scale human demonstrations. Starting with just 312 human-annotated computer use trajectories, we further improved data quality by synthesizing diverse action decisions with Claude 3.7 Sonnet. Trained on these enriched trajectories, our PC Agent-E model achieved a remarkable 141% relative improvement, surpassing the strong Claude 3.7 Sonnet with extended thinking on WindowsAgentArena-V2, an improved benchmark we also released. Furthermore, PC Agent-E demonstrates strong generalizability to different operating systems on OSWorld. Our findings suggest that strong computer use capabilities can be stimulated from a small amount of high-quality trajectory data.

</details>


<details>
<summary>
  <a href="https://arxiv.org/abs/2411.02337">WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning</a>
  <a href="https://github.com/THUDM/WebRL"><img src="https://img.shields.io/github/stars/THUDM/WebRL?style=social" alt="GitHub Stars"/></a>
</summary>

- Date: Nov, 2024  
- Env: Web  
- RL: Curriculum-RL  
- Base Model: Llama-3.1-8B, GLM-4-9B  
- Benchmark: WebArena-Lite  
- Reward: Rule-based  

**TLDR**:  
Large language models (LLMs) have shown remarkable potential as autonomous agents, particularly in web-based tasks. However, existing LLM web agents heavily rely on expensive proprietary LLM APIs, while open LLMs lack the necessary decision-making capabilities. This paper introduces WebRL, a self-evolving online curriculum reinforcement learning framework designed to train high-performance web agents using open LLMs. WebRL addresses three key challenges in building LLM web agents, including the scarcity of training tasks, sparse feedback signals, and policy distribution drift in online learning. Specifically, WebRL incorporates 1) a self-evolving curriculum that generates new tasks from unsuccessful attempts, 2) a robust outcome-supervised reward model (ORM), and 3) adaptive reinforcement learning strategies to ensure consistent improvements. We apply WebRL to transform open Llama-3.1 and GLM-4 models into proficient web agents. On WebArena-Lite, WebRL improves the success rate of Llama-3.1-8B from 4.8% to 42.4%, and from 6.1% to 43% for GLM-4-9B. These open models significantly surpass the performance of GPT-4-Turbo (17.6%) and GPT-4o (13.9%) and outperform previous state-of-the-art web agents trained on open LLMs (AutoWebGLM, 18.2%). Our findings demonstrate WebRL's effectiveness in bridging the gap between open and proprietary LLM-based web agents, paving the way for more accessible and powerful autonomous web interaction systems.

</details>

