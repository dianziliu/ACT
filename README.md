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

User-generated reviews often contain certain key aspects of many items and are particularly important for understanding users’ shopping interests. Recent recommendation methods utilize aspect graphs to represent the relationships between users and aspects and then apply graph neural networks to embed user preferences. However,
different users may choose different words or phrases to describe a certain aspect of an item, resulting in a sparse aspect graph. The sparse graph hinders the information propagation between users and aspects and fails to explore the user interest in potential aspects. In this paper, we propose an Aspect Category Transformer (ACT) for the aspect-based recommendation, which introduces aspect categories to complete the relationships between different aspect terms and mines the potentially relevant aspect terms. Potentially relevant aspect terms can be categorized into semantically similar aspect terms and semantically unrelated aspect terms. To effectively identify potentially semantically unrelated aspect terms, we design an aspect filter that considers graph features and user preferences to extract relevant aspect terms from the entire set of aspect terms. Meanwhile, to capture the relationships between semantically similar aspect terms, we introduce aspect categories, leveraging category information to establish connections among similar aspect terms.
Finally, we input the potential aspect terms and their corresponding aspect category features into the Transformer to explore the global relationships between users and aspect terms. Experiments on five real-world datasets demonstrate that the predictions of ACT significantly outperform SOTA baselines by 2.84% and 2.36% in MSE and MAE, respectively.

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
