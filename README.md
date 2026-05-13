<h1 align="center">
ACT
</h1>

<p align="center">
  <a href="#2-quick-start🚀">Quick Start</a> •
  <!-- <a href="https://dl.acm.org/doi/10.1145/3583780.3614770">Paper</a> • -->
  <!-- <a href="#3-citation☕️">Citation</a> -->
</p>

Repo for the under review paper of “ACT: Exploring Aspect Terms and Categories for Transformer-based Recommendationtion”

## 1. Introduction✨

User reviews provide rich aspect-level signals that are crucial for modeling user preferences. Existing aspect-based recommendation methods typically construct aspect graphs and apply graph neural networks to propagate information among users, items, and aspects. However, these approaches suffer from severe sparsity, as different users often use diverse terms to describe the same aspect, leading to fragmented aspect spaces and limited user–item overlap. Such sparsity not only hinders effective information propagation but also restricts the model’s ability to capture unmentioned yet potentially relevant aspects. To address these limitations, we propose ACT, an Aspect Category and Term fusion framework that jointly models coarse-grained aspect categories and fine-grained aspect terms for recommendation. Specifically, we first develop an Adaptive Cache Prompting Strategy (ACPS) to construct dynamically evolving aspect categories from extracted aspect terms, enabling scalable and semantically consistent category organization. Based on that, ACT introduces aspect categories as structural semantic anchors and design a category-enhanced aspect Transformer to explicitly model cross-level interactions between category-level semantic priors and term-level preference signals within a unified attention space. To further improve robustness and scalability, we introduce a relevance-aware aspect term selection method that leverages graph-structural signals and user preference cues to identify contextually relevant aspect terms, reducing noise while expanding the model’s effective receptive field. Experiments on five benchmark datasets demonstrate that ACT significantly outperforms state-of-the-art baselines by $3.12\%$ in mean absolute error and $2.32\%$ in Recall@5. More importantly, ACT can help identify potentially interesting aspects for users, as illustrated in our case study.

## 2. Quick Start🚀

1. File Structure

```
.\data [datasets]  
.\AspectCategoryExtraction
.\AspectTermExtration  [code for extracting aspect from datasets]  
.\src  
```

2. extract aspect terms and categories.

```sh
python AspectTermExtraction/1_aspext_extractor.py
python AspectTermExtraction/2_aspect_merge.py
python AspectTermExtraction/3_aspect_filter.py
python AspectTermExtraction/4_aspect_sentiment.py
python AspectCategoryExtraction/run.py
```

> [Note!] We use seta to calculate the review sentiment scores.
> The package of seta needs to be used at the root user.

3. training the model

```sh
python src/data_process.py
python src/data_gp.py
python src/train.py
```
