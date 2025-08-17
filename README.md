# Awesome Agentic Search and GUI Agents

This repository is a curated collection of research papers and resources on building next-generation GUI Agents and Agentic Search agents.

GUI Agents – Autonomous systems that can interact with graphical user interfaces by perceiving UI elements (e.g., buttons, inputs, menus) and performing human-like actions such as clicking, typing, and dragging to accomplish tasks.

Agentic Search / Deep Research – AI agents that autonomously plan, reason, and synthesize information across sources for complex tasks.


# Awesome Papers

## Table of Contents
  
1. [Agentic Search](#agentic-search)  
2. [GUI Agent](#gui-agent)


## Agentic Search

<details> <summary> <a href="https://arxiv.org/html/2506.15841v1">MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents</a> <a href="https://github.com/MIT-MI/MEM1"><img src="https://img.shields.io/github/stars/MIT-MI/MEM1?style=social" alt="GitHub Stars"/></a> </summary>

- Date: Jun, 2025 
- Env: Wikipedia dump, Google Search API, WebShop browser env 
- RL: PPO
- Base Model: Qwen2.5-7B
- Benchmark: HotpotQA+NQ (Augmented); WebShop 
- reward: Rule-based for QA; environment reward for WebShop 

**TLDR**:  Modern language agents must operate over long-horizon, multi-turn interactions, where they retrieve external information, adapt to observations, and answer interdependent queries. Yet, most LLM systems rely on full-context prompting, appending all past turns regardless of their relevance. This leads to unbounded memory growth, increased computational costs, and degraded reasoning performance on out-of-distribution input lengths. We introduce MEM1, an end-to-end reinforcement learning framework that enables agents to operate with constant memory across long multi-turn tasks. At each turn, MEM1 updates a compact shared internal state that jointly supports memory consolidation and reasoning. This state integrates prior memory with new observations from the environment while strategically discarding irrelevant or redundant information. To support training in more realistic and compositional settings, we propose a simple yet effective and scalable approach to constructing multi-turn environments by composing existing datasets into arbitrarily complex task sequences. Experiments across three domains, including internal retrieval QA, open-domain web QA, and multi-turn web shopping, show that MEM1-7B improves performance by 3.5× while reducing memory usage by 3.7× compared to Qwen2.5-14B-Instruct on a 16-objective multi-hop QA task, and generalizes beyond the training horizon. Our results demonstrate the promise of reasoning-driven memory consolidation as a scalable alternative to existing solutions for training long-horizon interactive agents, where both efficiency and performance are optimized.
</details>

<details> <summary> <a href="https://arxiv.org/abs/2507.02259">MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent</a> <a href="https://github.com/BytedTsinghua-SIA/MemAgent"><img src="https://img.shields.io/github/stars/BytedTsinghua-SIA/MemAgent?style=social" alt="GitHub Stars"/></a> </summary>

- Date: Jul, 2025 
- Env: Long-context text reading & QA (RULER-HotpotQA, NIAH, VT); no external tools required 
- RL: DAPO 
- Base Model: Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct
- Benchmark: RULER (QA part), Needle-in-a-Haystack
- reward: Rule-based

**TLDR**: 
Despite improvements by length extrapolation, efficient attention and memory modules, handling infinitely long documents with linear complexity without performance degradation during extrapolation remains the ultimate challenge in long-text processing. We directly optimize for long-text tasks in an end-to-end fashion and introduce a novel agent workflow, MemAgent, which reads text in segments and updates the memory using an overwrite strategy. We extend the DAPO algorithm to facilitate training via independent-context multi-conversation generation. MemAgent has demonstrated superb long-context capabilities, being able to extrapolate from an 8K context trained on 32K text to a 3.5M QA task with performance loss < 5% and achieves 95%+ in 512K RULER test.
</details>

<details>
<summary> <a href="https://arxiv.org/abs/2507.16727v1">Deliberative Searcher: Improving LLM Reliability via Reinforcement Learning with Constraints</a> </summary>

- Date: Jul, 2025 
- Env: Wikipedia dump, Google Search API
- RL: GRPO
- Base Model: 7B and 72B checkpoint (Shanghai AI Lab)  
- Benchmark: Multi-hop QA, GAIA, xBench-DeepSearch;
- reward: Rule-based

**TLDR**: 
Improving the reliability of large language models (LLMs) is critical for deploying them in real-world scenarios. In this paper, we propose Deliberative Searcher, the first framework to integrate certainty calibration with retrieval-based search for open-domain question answering. The agent performs multi-step reflection and verification over Wikipedia data and is trained with a reinforcement learning algorithm that optimizes for accuracy under a soft reliability constraint. Empirical results show that proposed method improves alignment between model confidence and correctness, leading to more trustworthy outputs. This paper will be continuously updated.

</details>


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


<details>
<summary>
  <a href="https://arxiv.org/abs/2503.09516">Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning</a>
  <a href="https://github.com/PeterGriffinJin/Search-R1"><img src="https://img.shields.io/github/stars/PeterGriffinJin/Search-R1?style=social" alt="GitHub Stars"/></a>
</summary>

- Date: Aug, 2025  
- Env: API 
- RL: GRPO, PPO
- Base Model: Qwen-2.5-3B (Base/Instruct), Qwen-2.5-7B (Base/Instruct) 
- Benchmark: Single-Hop QA, Multi-Hop QA
- reward: Rule-based  

**TLDR**:  
Efficiently acquiring external knowledge and up-to-date information is essential for effective reasoning and text generation in large language models (LLMs). Prompting advanced LLMs with reasoning capabilities to use search engines during inference is often suboptimal, as the LLM might not fully possess the capability on how to interact optimally with the search engine. This paper introduces Search-R1, an extension of reinforcement learning (RL) for reasoning frameworks where the LLM learns to autonomously generate (multiple) search queries during step-by-step reasoning with real-time retrieval. Search-R1 optimizes LLM reasoning trajectories with multi-turn search interactions, leveraging retrieved token masking for stable RL training and a simple outcome-based reward function. Experiments on seven question-answering datasets show that Search-R1 improves performance by 41% (Qwen2.5-7B) and 20% (Qwen2.5-3B) over various RAG baselines under the same setting. This paper further provides empirical insights into RL optimization methods, LLM choices, and response length dynamics in retrieval-augmented reasoning. The code and model checkpoints are available at this https URL. 
</details>

<details>
<summary>
  <a href="https://www.arxiv.org/abs/2507.19849">Agentic Reinforced Policy Optimization</a>
  <a href="https://github.com/dongguanting/ARPO"><img src="https://img.shields.io/github/stars/dongguanting/ARPO?style=social" alt="GitHub Stars"/></a>
</summary>

- Date: Jul, 2025  
- Env: API and Python
- RL: ARPO with Cold Start
- Base Model: Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct, Llama3.1-8B-Instruct
- Benchmark: AIME, WebWalkerQA, GAIA, HLE, xbench, Multi-Hop QA
- reward: Rule-based

**TLDR**:  
Large-scale reinforcement learning with verifiable rewards (RLVR) has demonstrated its effectiveness in harnessing the potential of large language models (LLMs) for single-turn reasoning tasks. In realistic reasoning scenarios, LLMs can often utilize external tools to assist in task-solving processes. However, current RL algorithms inadequately balance the models' intrinsic long-horizon reasoning capabilities and their proficiency in multi-turn tool interactions. To bridge this gap, we propose Agentic Reinforced Policy Optimization (ARPO), a novel agentic RL algorithm tailored for training multi-turn LLM-based agents. Through preliminary experiments, we observe that LLMs tend to exhibit highly uncertain behavior, characterized by an increase in the entropy distribution of generated tokens, immediately following interactions with external tools. Motivated by this observation, ARPO incorporates an entropy-based adaptive rollout mechanism, dynamically balancing global trajectory sampling and step-level sampling, thereby promoting exploration at steps with high uncertainty after tool usage. By integrating an advantage attribution estimation, ARPO enables LLMs to internalize advantage differences in stepwise tool-use interactions. Our experiments across 13 challenging benchmarks in computational reasoning, knowledge reasoning, and deep search domains demonstrate ARPO's superiority over trajectory-level RL algorithms. Remarkably, ARPO achieves improved performance using only half of the tool-use budget required by existing methods, offering a scalable solution for aligning LLM-based agents with real-time dynamic environments.
</details>

<details>
<summary>
  <a href="https://arxiv.org/abs/2507.15061">WebShaper: Agentically Data Synthesizing via Information-Seeking Formalization</a>
  <a href="https://github.com/Alibaba-NLP/WebAgent"><img src="https://img.shields.io/github/stars/Alibaba-NLP/WebAgent?style=social" alt="GitHub Stars"/></a>
</summary>

- Date: Jul, 2025  
- Env: API 
- RL: GRPO with Cold Start
- Base Model: QwQ-32B, Qwen-2.5-32B, Qwen-2.5-72B.
- Benchmark: WebWalkerQA, GAIA
- reward: Rule-based

**TLDR**:  
The advent of Large Language Model (LLM)-powered agents has revolutionized artificial intelligence by enabling solutions to complex, open-ended tasks through web-based information-seeking (IS) capabilities. The scarcity of high-quality training data has limited the development of IS agents. Existing approaches typically adopt an information-driven paradigm that first collects web data and then generates questions based on the retrieval. However, this may lead to inconsistency between information structure and reasoning structure, question and answer. To mitigate, we propose a formalization-driven IS data synthesis framework WebShaper to construct a dataset. WebShaper systematically formalizes IS tasks through set theory. Central to the formalization is the concept of Knowledge Projections (KP), which enables precise control over reasoning structure by KP operation compositions. During synthesis, we begin by creating seed tasks, then use a multi-step expansion process. At each step, an agentic Expander expands the current formal question more complex with retrieval and validation tools based on our formalization. We train our model on the synthesized dataset. Experiment results demonstrate that WebShaper achieves state-of-the-art performance among open-sourced IS agents on GAIA and WebWalkerQA benchmarks.
</details>

<details>
<summary>
  <a href="https://arxiv.org/abs/2507.02592">WebSailor: Navigating Super-human Reasoning for Web Agent</a>
  <a href="https://github.com/Alibaba-NLP/WebAgent"><img src="https://img.shields.io/github/stars/Alibaba-NLP/WebAgent?style=social" alt="GitHub Stars"/></a>
</summary>

- Date: Jul, 2025  
- Env: API 
- RL: GRPO with Cold Start
- Base Model: Qwen-2.5-3B, Qwen-2.5-7B, Qwen-2.5-32B, Qwen-2.5-72B.
- Benchmark: BrowseComp, Xbench-DeepSearch, GAIA
- reward: Rule-based 

**TLDR**:  
Transcending human cognitive limitations represents a critical frontier in LLM training. Proprietary agentic systems like DeepResearch have demonstrated superhuman capabilities on extremely complex information-seeking benchmarks such as BrowseComp, a feat previously unattainable. We posit that their success hinges on a sophisticated reasoning pattern absent in open-source models: the ability to systematically reduce extreme uncertainty when navigating vast information landscapes. Based on this insight, we introduce WebSailor, a complete post-training methodology designed to instill this crucial capability. Our approach involves generating novel, high-uncertainty tasks through structured sampling and information obfuscation, RFT cold start, and an efficient agentic RL training algorithm, Duplicating Sampling Policy Optimization (DUPO). With this integrated pipeline, WebSailor significantly outperforms all opensource agents in complex information-seeking tasks, matching proprietary agents' performance and closing the capability gap.
</details>

<details>
<summary>
  <a href="https://arxiv.org/abs/2505.22501">EvolveSearch: An Iterative Self-Evolving Search Agent</a>
</summary>

- Date: May, 2025  
- Env: API 
- RL: Iterative self-evolution with SFT and GRPO
- Base Model: Qwen2.5-7B-Instruct
- Benchmark: Single-Hop QA, Multi-Hop QA
- reward: Rule-based 

**TLDR**:  
The rapid advancement of large language models (LLMs) has transformed the landscape of agentic information seeking capabilities through the integration of tools such as search engines and web browsers. However, current mainstream approaches for enabling LLM web search proficiency face significant challenges: supervised fine-tuning struggles with data production in open-search domains, while RL converges quickly, limiting their data utilization efficiency. To address these issues, we propose EvolveSearch, a novel iterative self-evolution framework that combines SFT and RL to enhance agentic web search capabilities without any external human-annotated reasoning data. Extensive experiments on seven multi-hop question-answering (MHQA) benchmarks demonstrate that EvolveSearch consistently improves performance across iterations, ultimately achieving an average improvement of 4.7\% over the current state-of-the-art across seven benchmarks, opening the door to self-evolution agentic capabilities in open web search domains.
</details>

<details>
<summary>
  <a href="https://arxiv.org/abs/2504.03160">DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments</a>
  <a href="https://github.com/GAIR-NLP/DeepResearcher"><img src="https://img.shields.io/github/stars/GAIR-NLP/DeepResearcher?style=social" alt="GitHub Stars"/></a>
</summary>

- Date: Apr, 2025  
- Env: API
- RL: GRPO
- Base Model: Qwen2.5-7B-Instruct
- Benchmark: Single-Hop QA, Multi-Hop QA
- reward: Rule-based 

**TLDR**:  
Large Language Models (LLMs) equipped with web search capabilities have demonstrated impressive potential for deep research tasks. However, current approaches predominantly rely on either manually engineered prompts (prompt engineering-based) with brittle performance or reinforcement learning within controlled Retrieval-Augmented Generation (RAG) environments (RAG-based) that fail to capture the complexities of real-world interaction. In this paper, we introduce DeepResearcher, the first comprehensive framework for end-to-end training of LLM-based deep research agents through scaling reinforcement learning (RL) in real-world environments with authentic web search interactions. Unlike RAG-based approaches that assume all necessary information exists within a fixed corpus, our method trains agents to navigate the noisy, unstructured, and dynamic nature of the open web. We implement a specialized multi-agent architecture where browsing agents extract relevant information from various webpage structures and overcoming significant technical challenges. Extensive experiments on open-domain research tasks demonstrate that DeepResearcher achieves substantial improvements of up to 28.9 points over prompt engineering-based baselines and up to 7.2 points over RAG-based RL agents. Our qualitative analysis reveals emergent cognitive behaviors from end-to-end RL training, including the ability to formulate plans, cross-validate information from multiple sources, engage in self-reflection to redirect research, and maintain honesty when unable to find definitive answers. Our results highlight that end-to-end training in real-world web environments is not merely an implementation detail but a fundamental requirement for developing robust research capabilities aligned with real-world applications.
</details>

<details> <summary> <a href="https://arxiv.org/abs/2504.21776">WebThinker: Empowering Large Reasoning Models with Deep Research Capability</a> <a href="https://github.com/RUC-NLPIR/WebThinker"><img src="https://img.shields.io/github/stars/RUC-NLPIR/WebThinker?style=social" alt="GitHub Stars"/></a> </summary>

- Date: Apr, 2025. 
- Env: Browser
- RL: Online DPO
- Base Model: QwQ-32B and DeepSeek-R1 distilled
- Benchmark: GPQA, GAIA, WebWalkerQA, HLE. 
- reward: Preferences built from reasoning correctness, tool usage, and final outputs.

**TLDR**: Large reasoning models (LRMs), such as OpenAI-o1 and DeepSeek-R1, demonstrate impressive long-horizon reasoning capabilities. However, their reliance on static internal knowledge limits their performance on complex, knowledge-intensive tasks and hinders their ability to produce comprehensive research reports requiring synthesis of diverse web information. To address this, we propose WebThinker, a deep research agent that empowers LRMs to autonomously search the web, navigate web pages, and draft research reports during the reasoning process. WebThinker integrates a Deep Web Explorer module, enabling LRMs to dynamically search, navigate, and extract information from the web when encountering knowledge gaps. It also employs an Autonomous Think-Search-and-Draft strategy, allowing the model to seamlessly interleave reasoning, information gathering, and report writing in real time. To further enhance research tool utilization, we introduce an RL-based training strategy via iterative online Direct Preference Optimization (DPO). Extensive experiments on complex reasoning benchmarks (GPQA, GAIA, WebWalkerQA, HLE) and scientific report generation tasks (Glaive) demonstrate that WebThinker significantly outperforms existing methods and strong proprietary systems. Our approach enhances LRM reliability and applicability in complex scenarios, paving the way for more capable and versatile deep research systems. 
</details>


<details> <summary> <a href="https://arxiv.org/abs/2505.16834">SimpleDeepSearcher: Deep Information Seeking via Web-Powered Reasoning Trajectory Synthesis</a> <a href="https://github.com/RUCAIBox/SimpleDeepSearcher"><img src="https://img.shields.io/github/stars/RUCAIBox/SimpleDeepSearcher?style=social" alt="GitHub Stars"/></a> </summary>

- Date: May, 2025  
- Env: Search API 
- SFT only on 871 samples 
- Base Model: Qwen-2.5-7B, Qwen-2.5-32B, DeepSeek-Distilled-Qwen-2.5-32B, QwQ-32B. 
- Benchmark: Multi-Hop QA, FRAMES, GAIA 

**TLDR**: Retrieval-augmented generation (RAG) systems have advanced large language models (LLMs) in complex deep search scenarios requiring multi-step reasoning and iterative information retrieval. However, existing approaches face critical limitations that lack high-quality training trajectories or suffer from the distributional mismatches in simulated environments and prohibitive computational costs for real-world deployment. This paper introduces SimpleDeepSearcher, a lightweight yet effective framework that bridges this gap through strategic data engineering rather than complex training paradigms. Our approach synthesizes high-quality training data by simulating realistic user interactions in live web search environments, coupled with a multi-criteria curation strategy that optimizes the diversity and quality of input and output side. Experiments on five benchmarks across diverse domains demonstrate that SFT on only 871 curated samples yields significant improvements over RL-based baselines. Our work establishes SFT as a viable pathway by systematically addressing the data-scarce bottleneck, offering practical insights for efficient deep search systems. 

</details>

<details>
<summary>
  <a href="https://arxiv.org/abs/2505.04588">ZeroSearch: Incentivize the Search Capability of LLMs without Searching</a>
  <a href="https://github.com/Alibaba-NLP/ZeroSearch"><img src="https://img.shields.io/github/stars/Alibaba-NLP/ZeroSearch?style=social" alt="GitHub Stars"/></a>
</summary>

- Date: May, 2025  
- Env: LLM Synthesized
- RL: REINFORCE, PPO, GRPO
- Base Model: Qwen-2.5-3B(Base/Instruct), Qwen-2.5-7B (Base/Instruct), Llama-3.2-3B (Base/Instruct)
- Benchmark: Single-Hop QA, Multi-Hop QA

**TLDR**: 
Effective information searching is essential for enhancing the reasoning and generation capabilities of large language models (LLMs). Recent research has explored using reinforcement learning (RL) to improve LLMs' search capabilities by interacting with live search engines in real-world environments. While these approaches show promising results, they face two major challenges: (1) Uncontrolled Document Quality: The quality of documents returned by search engines is often unpredictable, introducing noise and instability into the training process. (2) Prohibitively High API Costs: RL training requires frequent rollouts, potentially involving hundreds of thousands of search requests, which incur substantial API expenses and severely constrain scalability. To address these challenges, we introduce ZeroSearch, a novel RL framework that incentivizes the capabilities of LLMs to use a real search engine with simulated searches during training. Our approach begins with lightweight supervised fine-tuning to transform the LLM into a retrieval module capable of generating both useful and noisy documents in response to a query. During RL training, we employ a curriculum-based rollout strategy that incrementally degrades the quality of generated documents, progressively eliciting the model's reasoning ability by exposing it to increasingly challenging retrieval scenarios. Extensive experiments demonstrate that ZeroSearch effectively incentivizes the search capabilities of LLMs using a 3B LLM as the retrieval module. Remarkably, a 7B retrieval module achieves comparable performance to the real search engine, while a 14B retrieval module even surpasses it. Furthermore, it generalizes well across both base and instruction-tuned models of various parameter sizes and is compatible with a wide range of RL algorithms.


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
  <a href="https://arxiv.org/abs/2506.03533">Go-Browse: Training Web Agents with Structured Exploration</a>
  <a href="https://github.com/ApGa/Go-Browse"><img src="https://img.shields.io/github/stars/ApGa/Go-Browse?style=social" alt="GitHub Stars"/></a>
</summary>

- Date: Jun, 2025  
- Env: Web  
- Base Model: Qwen-2.5-7B-Instruct
- Benchmark: WebArena  

**TLDR**:  
One of the fundamental problems in digital agents is their lack of understanding of their environment. For instance, a web browsing agent may get lost in unfamiliar websites, uncertain what pages must be visited to achieve its goals. To address this, we propose Go-Browse, a method for automatically collecting diverse and realistic web agent data at scale through structured exploration of web environments. Go-Browse achieves efficient exploration by framing data collection as a graph search, enabling reuse of information across exploration episodes. We instantiate our method on the WebArena benchmark, collecting a dataset of 10K successful task-solving trajectories and 40K interaction steps across 100 URLs. Fine-tuning a 7B parameter language model on this dataset achieves a success rate of 21.7% on the WebArena benchmark, beating GPT-4o mini by 2.4% and exceeding current state-of-the-art results for sub-10B parameter models by 2.9%.

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

