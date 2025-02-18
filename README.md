<p align="center">
  <h1 align="center">🔥 Awesome-LLM-Ensemble 

"Harnessing Multiple Large Language Models: A Survey on LLM Ensemble"</h2>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=3slpkWAAAAAJ&hl=zh-CN">Zhijun Chen</a>,
    <a href="https://scholar.google.com/citations?user=6v6JfdsAAAAJ&hl=zh-CN">Jingzheng Li</a>,
    <a href="https://scholar.google.com/citations?user=mzXg1s8AAAAJ&hl=zh-CN">Pengpeng Chen</a>,
    <a href="https://scholar.google.com/citations?user=Hh-IHMoAAAAJ&hl=zh-CN">Zhuoran Li</a>,
    <a href="https://scholar.google.com/citations?user=buUlnJUAAAAJ&hl=zh-CN">Kai Sun</a>,
    <a href="https://luoyk1999.github.io/">Yuankai Luo</a>,
    <a href="https://scholar.google.com/citations?hl=zh-CN&user=PnDqlPkAAAAJ&view_op=list_works&sortby=pubdate">Qianren Mao</a>,
    <a href="https://scholar.google.co.uk/citations?user=fnfg9S0AAAAJ&hl=en">Dingqi Yang</a>,
    <a href="https://scholar.google.com/citations?user=HWOWCdcAAAAJ&hl=zh-CN">Hailong Sun</a>,
    <a href="https://scholar.google.com/citations?user=D0lL1r0AAAAJ&hl=zh-CN/">Philip S. Yu</a>
  </p>
  <p align="center">
    <img src="fig/logo.png" alt="Logo" width="50%">
  </p>
</p>

<!-- Make the "Maintained?" and License info centered -->
<div align="center">
   <p>
      <a href="https://github.com/junchenzhi/Awesome-LLM-Ensemble/commits/main">
        <img src="https://img.shields.io/badge/Maintained%3F-yes-green.svg" alt="Maintenance">
      </a>
      <a href="https://github.com/junchenzhi/Awesome-LLM-Ensemble/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License">
      </a>
      <a href="https://awesome.re">
        <img src="https://awesome.re/badge.svg" alt="Awesome">
      </a>
      <a href="https://github.com/junchenzhi/Awesome-LLM-Ensemble/stargazers">
        <img src="https://img.shields.io/github/stars/junchenzhi/Awesome-LLM-Ensemble?color=green" alt="GitHub stars">
      </a>
      <a href="https://github.com/junchenzhi/Awesome-LLM-Ensemble/network">
        <img src="https://img.shields.io/github/forks/junchenzhi/Awesome-LLM-Ensemble?color=blue&label=Forks" alt="GitHub forks">
      </a>
  </p>
</div>



<h5 align="center">If you like our project, please give it a star ⭐ to show your support. 


For this emerging topic, we hope this project can provide some reference for researchers and look forward to more interesting studies!
</h5>



# 📣 News and Notices
> 🔥🔥🔥 This is a collection of papers on  ***LLM Ensemble***.   
It's based on our survey paper: Harnessing Multiple Large Language Models: A Survey on LLM Ensemble. 

> **[Always]** We will try to make this list updated frequently. If you found any error or any missed/new paper, please don't hesitate to contact [us](zhijunchen@buaa.edu.cn).

> **[2025/02/19]** We will release our paper on arXiv in the next few days. Stay tuned.




---



&nbsp; 
&nbsp;  
&nbsp; 




- [Contents](#Awesome-LLM-Ensemble)
  - 1 [LLM Ensemble and Taxonomy](#1-llm-ensemble-and-taxonomy)
    - 1.1 [LLM Ensemble](#11-llm-ensemble)
    - 1.2 [Taxonomy](#12-taxonomy)
  - 2 [Papers](#2-papers)
    - 2.1 [Ensemble Before Inference](#21-ensemble-before-inference)
    - 2.2 [Ensemble During Inference](#22-ensemble-during-inference)
    - 2.3 [Ensemble After Inference](#23-ensemble-after-inference)
    - 2.4 [Others: Benchmarks and Applications](#24-others-benchmarks-and-applications)
  - 3 [Summarization](#3-summarization)





# 1. LLM Ensemble and Taxonomy

## 1.1 LLM Ensemble
**Paper Abstract**:

LLM Ensemble---which involves the comprehensive use of multiple large language models (LLMs), each aimed at handling user queries during the downstream inference, to benefit from their individual strengths---has gained substantial attention recently. 
The widespread availability of LLMs, coupled with their varying strengths and out-of-the-box usability, has profoundly advanced the field of LLM Ensemble. 
This paper presents the first systematic review of recent developments in LLM Ensemble. 
First, we introduce our taxonomy of LLM Ensemble and discuss several related research problems. 
Then, we provide a more in-depth classification of methods under the broad categories of ``ensemble-before-inference, ensemble-during-inference, ensemble-after-inference'', and review all relevant methods. Finally, we introduce related benchmarks and applications, summarize existing studies, and suggest several future research directions. 
A curated list of papers  on LLM Ensemble is available at https://github.com/junchenzhi/Awesome-LLM-Ensemble.

## 1.2 Taxonomy

<div align=center>
<img src="./fig/illustration.png" width="95%">

**Figure 1:  Illustration of LLM Ensemble Taxonomy.** (Note that for *(b) ensemble-during-inference* paradigm, there is also a *process-level ensemble* approach that we have not represented in the figure, mainly because that this approach is instantiated by a single method.)
</div>



<div align=center>
<img src="./fig/all_methods.png" width="95%">

**Figure 2: Taxonomy of All LLM Ensemble Methods.**
</div>





- ***(a) Ensemble before inference.***  
In essence, this approach employs a routing algorithm prior to LLM inference to allocate a specific query to the most suitable model, allowing the selected model that is specialized for the query and typically more cost-efficient inference to perform the task.
 Existing methods can be classified into two categories, depending on whether the router necessitates the use of pre-customized data for pre-training:   
  - ***(a,1) Pre-training router;***
  - ***(a,2) Non pre-training router.***




- ***(b) Ensemble during inference.***  
As the most granular form of ensemble among the three broad categories, this type of approach encompasses: 
  - ***(b,1) Token-level ensemble*** methods, which integrate the token-level outputs of multiple models at the finest granularity of decoding;
  - ***(b,2) Span-level ensemble*** methods, which conduct ensemble at the level of a sequence fragment (e.g., a span of four words); 
  - ***(b,3) Process-level ensemble*** methods, which select the optimal reasoning process step-by-step within the reasoning chain for a given complex reasoning task. 
Note that for these ensemble-during-inference methods, the aggregated text segments will be concatenated with the previous text and fed again to models.



- ***(c) Ensemble after inference.***  
These methods can be classified into two categories:
  - ***(c,1) Non cascade*** methods, which perform ensemble using multiple complete responses contributed from all LLM candidates;
  - ***(c,2) Cascade*** methods, which consider both performance and inference costs, progressively reasoning through a chain of LLM candidates largely sorted by model size to find the most suitable inference response.




---



&nbsp; 
&nbsp;  
&nbsp; 



# 2. Papers

## 2.1 Ensemble Before Inference


<div align=center>
<img src="./fig/before.png" width="90%">

Figure 3:  Summary analysis of the key attributes of ensemble-before-inference methods.
</div>




### 2.1.1 (a,1) Pre-Trained Router

| Year |                                                           Title                                                            |    Name    |                             Code                              | 
|:----:|:--------------------------------------------------------------------------------------------------------------------------:|:----------:|:-------------------------------------------------------------:|
| 2023 |                     [LLM Routing with Benchmark Datasets](https://openreview.net/forum?id=k9EfAJhFZc)                      |     -      |                               -                               |
| 2024 |                 [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/abs/2406.18665)                  |  RouteLLM  |        [Official](https://github.com/lm-sys/RouteLLM)         |
| 2024 |              [Hybrid LLM: Cost-Efficient and Quality-Aware Query Routinga](https://arxiv.org/abs/2404.14618)               | Hybrid-LLM |  [Official](https://github.com/m365-core/hybrid_llm_routing)  |
| 2025 |  [LLM Bandit: Cost-Efficient LLM Generation via Preference-Conditioned Dynamic Routing](https://arxiv.org/abs/2502.02743)  |     -      |  [Official](https://github.com/m365-core/hybrid_llm_routing)  |
| 2024 |        [Harnessing the Power of Multiple Minds: Lessons Learned from LLM Routing](https://arxiv.org/abs/2405.00467)        |    -       |  [Official](https://github.com/kvadityasrivatsa/llm-routing)  |
| 2024 |   [MetaLLM: A High-performant and Cost-efficient Dynamic Framework for Wrapping LLMs](https://arxiv.org/abs/2407.10834)    |  MetaLLM   | [Official](https://github.com/mail-research/MetaLLM-wrapper/) |
| 2024 |     [SelectLLM: Query-Aware Efficient Selection Algorithm for Large Language Models](https://arxiv.org/abs/2408.08545)     | SelectLLM  |                               -                               |
| 2024 |           [Bench-CoE: a Framework for Collaboration of Experts from Benchmark](https://arxiv.org/abs/2412.04167)           | Bench-CoE  |      [Official](https://github.com/ZhangXJ199/Bench-CoE)      |
| 2024 |    [Routing to the Expert: Efficient Reward-guided Ensemble of Large Language Models](https://arxiv.org/abs/2311.08692)    |   ZOOTER   |                               -                               |
| 2024 |          [TensorOpera Router: A Multi-Model Router for Efficient LLM Inference](https://arxiv.org/abs/2408.12320)          | TO-Router  |                               -                               |
| 2024 |       [Query Routing for Homogeneous Tools: An Instantiation in the RAG Scenario](https://arxiv.org/abs/2406.12429)        | HomoRouter |                               -                               |
| 2023 |       [Fly-Swat or Cannon? Cost-Effective Language Model Choice via Meta-Modeling](https://arxiv.org/abs/2308.06077)        |    FORC    |         [Official](https://github.com/epfl-dlab/forc)         |
| 2024 |       [Routoo: Learning to Route to Large Language Models Effectively](https://arxiv.org/abs/2401.13979)        |   Routoo   |                               -                               |



### 2.1.2 (a,2) Non pre-trained router

| Year |                                                                      Title                                                                       |   Name    | Code | 
|:----:|:------------------------------------------------------------------------------------------------------------------------------------------------:|:---------:|:----:|
| 2024 |  [PickLLM: Context-Aware RL-Assisted Large Language Model Routing](https://arxiv.org/abs/2412.12170)   |     PickLLM      |  -   |
| 2024 |  [Eagle: Efficient Training-Free Router for Multi-LLM Inference](https://arxiv.org/abs/2409.15518)   | Eagle |  -   |
| 2024 |  [Blending Is All You Need: Cheaper, Better Alternative to Trillion-Parameters LLM](https://arxiv.org/abs/2401.02994)   | Blending  |  -   |




&nbsp; 

## 2.2 Ensemble During Inference

<div align=center>
<img src="./fig/during.png" width="90%">

Figure 4:  Summary analysis of the key attributes of ensemble-during-inference methods.
</div>



### 2.2.1 (b,1) Token-Level Ensemble


| Year |                                                                      Title                                                                       |     Name     |                     Code                     | 
|:----:|:------------------------------------------------------------------------------------------------------------------------------------------------:|:------------:|:--------------------------------------------:|
| 2024 |  [Breaking the Ceiling of the LLM Community by Treating Token Generation as a Classification for Ensembling](https://arxiv.org/abs/2406.12585)   |     GaC      |  [Official](https://github.com/yaoching0/GaC)   |
| 2024 |     [Ensemble Learning for Heterogeneous Large Language Models with Deep Parallel Collaboration](https://openreview.net/forum?id=7arAADUK6D)     |    DeePEn    |  [Official](https://github.com/JieyuZ2/wrench)  |
| 2024 |                       [Bridging the Gap between Different Vocabularies for LLM Ensemble](https://arxiv.org/abs/2404.09492)                       |     EVA      |   [Official](https://github.com/xydaytoy/EVA)   |
| 2024 |            [Determine-Then-Ensemble: Necessity of Top-k Union for Large Language Model Ensembling](https://arxiv.org/abs/2410.03777)             |    UniTe     |                        -                        |
| 2024 |                     [Pack of LLMs: Model Fusion at Test-Time via Perplexity Optimization](https://arxiv.org/abs/2404.11531)                      |   PackLLM    |  [Official](https://github.com/cmavro/PackLLM)  |
| 2025 |   [CITER: Collaborative Inference for Efficient Large Language Model Decoding with Token-Level Routing](https://arxiv.org/abs/2502.01976)        |  CITER       | [Official](https://github.com/aiming-lab/CITER) |



### 2.2.2 (b,2) Span-Level Ensemble

| Year |                                                                      Title                                                                       | Name | Code | 
|:----:|:------------------------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|
| 2024 |  [Cool-Fusion: Fuse Large Language Models without Training](https://arxiv.org/abs/2407.19807)   |   Cool-Fusion   |  -   |
| 2024 |  [Hit the Sweet Spot! Span-Level Ensemble for Large Language Models](https://arxiv.org/abs/2409.18583)   |   SweetSpan   |  -   |
| 2024 |  [SpecFuse: Ensembling Large Language Models via Next-Segment Prediction](https://arxiv.org/abs/2412.07380)   |   SpecFuse   |  -   |


### 2.2.3 (b,3) Process-Level Ensemble

| Year |                                                                      Title                                                                       | Name | Code | 
|:----:|:------------------------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|
| 2024 |  [Ensembling Large Language Models with Process Reward-Guided Tree Search for Better Complex Reasoning](https://arxiv.org/abs/2412.15797)   |   LE-MCTS  |  -   |


&nbsp; 


## 2.3 Ensemble After Inference


<div align=center>
<img src="./fig/after.png" width="90%">

Figure 5:  Summary analysis of the key attributes of ensemble-during-inference methods.
</div>


### 2.3.1 (c,1) Non Cascade

| Year |                                                                   Title                                                                   |     Name     |                               Code                                | 
|:----:|:-----------------------------------------------------------------------------------------------------------------------------------------:|:------------:|:-----------------------------------------------------------------:|
| 2024 |                                      [More Agents Is All You Need](https://arxiv.org/abs/2402.05120)                                      | Agent-Forest | [Official](https://github.com/MoreAgentsIsAllYouNeed/AgentForest) |
| 2024 |                              [Smoothie: Label Free Language Model Routing](https://arxiv.org/abs/2412.04692)                              |   SMOOTHIE   |       [Official](https://github.com/HazyResearch/smoothie)        |
| 2023 |                    [Getting MoRE out of Mixture of Language Model Reasoning Experts](https://arxiv.org/abs/2305.14628)                    |     MoRE     |            [Official](https://github.com/NoviScl/MoRE)            |
| 2023 |       [LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion](https://arxiv.org/abs/2306.02561)       | LLM-Blender  |       [Official](https://github.com/yuchenlin/LLM-Blender)        |
| 2024 |                       [LLM-TOPLA: Efficient LLM Ensemble by Maximising Diversity](https://arxiv.org/abs/2410.03953)                       |  LLM-TOPLA   |         [Official](https://github.com/git-disl/llm-topla)         |
| 2024 |    [URG: A Unified Ranking and Generation Method for Ensembling Language Models](https://aclanthology.org/2024.findings-acl.261/)         |    URG       |                                          -                        |






### 2.3.2 (c,2) Cascade

| Year |                                                                                    Title                                                                                     |                   Name                    |                           Code                          | 
|:----:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------:|:-------------------------------------------------------:|
| 2023 |                                     [EcoAssistant: Using LLM Assistant More Affordably and Accurately](https://arxiv.org/abs/2310.03046)                                     |               EcoAssistant                |    [Official](https://github.com/JieyuZ2/EcoAssistant)  |
| 2024 |                   [Large Language Model Cascades with Mixture of Thoughts Representations for Cost-efficient Reasoning](https://arxiv.org/abs/2310.03094)                    |                     -                     | [Official](https://github.com/MurongYue/LLM_MoT_cascade) |
| 2022 |                            [Model Cascading: Towards Jointly Improving Efficiency and Accuracy of NLP Systems](https://arxiv.org/abs/2210.05528)                             |              Model Cascading              |                             -                           |
| 2023 |                                      [Cache & Distil: Optimising API Calls to Large Language Models](https://arxiv.org/abs/2310.13561)                                       |              neural caching               | [Official](https://github.com/guillemram97/neural-caching) |
| 2023 |                                           [A Unified Approach to Routing and Cascading for LLMs](https://arxiv.org/abs/2410.10347)                                           |              Cascade Routing              |   [Official](https://github.com/eth-sri/cascade-routing)  |
| 2023 | [When Does Confidence-Based Cascade Deferral Suffice?](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1f09e1ee5035a4c3fe38a5681cae5815-Abstract-Conference.html) |                     -                     |                             -                             |
| 2023 |                        [FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance](https://arxiv.org/abs/2305.05176)                         |                 FrugalGPT                 |     -          |
| 2024 |                                       [Language Model Cascades: Token-level uncertainty and beyond](https://arxiv.org/abs/2404.10136)                                        |                     -                     |     -          |
| 2023 |                                              [AutoMix: Automatically Mixing Language Models](https://arxiv.org/abs/2310.12963)                                               |                  AutoMix                  |     -           |
| 2024 |                      [Dynamic Ensemble Reasoning for LLM Experts](https://arxiv.org/abs/2412.07448)                                               |          DER                              |     -           |



&nbsp; 


## 2.4 Others: Benchmarks and Applications

### 2.4.1 Benchmarks

| Year |                                                             Title                                                             | Benchmark Name |   Evaluation Goal    |                                  Code                                  | 
|:----:|:-----------------------------------------------------------------------------------------------------------------------------:|:--------------:|:--------------------:|:----------------------------------------------------------------------:|
| 2023 | [LLM-BLENDER: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion](https://arxiv.org/abs/2306.02561) |  MixInstruct   |     Performance      |             [Official](https://yuchenlin.xyz/LLM-Blender/)             |
| 2024 |        [RouterBench: A Benchmark for Multi-LLM Routing System](https://arxiv.org/abs/2403.12031)                              |  RouterBench   | Performance and cost |         [Official](https://github.com/withmartian/routerbench)         |





### 2.4.2 Applications


Beyond the methods presented before, the concept of LLM Ensemble has found applications in a variety of more specialized tasks and domains.
Here we give some examples:

| Year |                                                                                 Title                                                                                 |            Name            |                Task                |                                  Code                                  | 
|:----:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------:|:----------------------------------:|:----------------------------------------------------------------------:|
| 2023 |                     [Ensemble-Instruct: Generating Instruction-Tuning Data with a Heterogeneous Mixture of LMs](https://arxiv.org/pdf/2310.13961)                     |     Ensemble-Instruct      | Instruction-Tuning Data Generation |          [Official](https://github.com/IBM/ensemble-instruct)          |
| 2024 |                                  [Bayesian Calibration of Win Rate Estimation with LLM Evaluators](https://arxiv.org/abs/2411.04424)                                  | BWRS, Bayesian Dawid-Skene |        Win Rate Estimation         | [Official](https://github.com/yale-nlp/bay-calibration-llm-evaluators) |
| 2024 |     [PromptMind Team at MEDIQA-CORR 2024: Improving Clinical Text Correction with Error Categorization and LLM Ensembles](https://arxiv.org/abs/2405.08373)           |             -              |       SQL generation               |                                   -                                    |


---

&nbsp; 
&nbsp;  
&nbsp; 

## 3 Summarization

<div align=center>
<img src="./fig/summary.png" width="90%">

Figure 6:  Summary analysis of the key attributes of LLM Ensemble approaches.
</div>
