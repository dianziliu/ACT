<h1 align="center">
ACT
</h1>

<p align="center">
  <a href="#2-quick-start🚀">Quick Start</a> •
  <!-- <a href="https://dl.acm.org/doi/10.1145/3583780.3614770">Paper</a> • -->
  <!-- <a href="#3-citation☕️">Citation</a> -->
</p>



Repo for a under review paper of AAAI 2025: [ACT: Exploring Aspect Terms and Categories for
Transformer-based Recommendationtion]

## 1. Introduction✨
ser-generated reviews often contain certain key aspects of many items and are particularly important for understanding users’ shopping interests. Recent recommendation methods utilize aspect graphs to represent the relationships between users and aspects and then apply graph neural networks to embed user preferences. However,
different users may choose different words or phrases to describe a certain aspect of an item, resulting in a sparse aspect graph. The sparse graph hinders the information propagation between users and aspects and fails to explore the user interest in potential aspects. In this paper, we propose an Aspect Category Transformer (ACT) for the aspect-based recommendation, which explores the feasibility of replacing GNNs with a more powerful Transformer
architecture to capture interactions between users, items, and aspects. Since it is infeasible to train Transformer on large graphs, we first design an aspect filter to select a comprehensive set of explicitly mentioned and implicitly relevant aspects from the entire set for each user and item. Then, considering aspect categories refer
to the categories that aspect terms belong to and usually have a small range, we introduce aspect categories into the Transformer, enabling the model to further capture relationships between aspect terms and broader contextual information. Finally, through the aspect filter and the aspect category-enhanced Transformer, ACT extends user preference from their interacted aspects to non-interacted but potentially interesting aspects. Experiments on five
real-world datasets demonstrate that the predictions of ACT significantly outperform SOTA baselines by 2.84% and 2.36% in MSEand MAE, respectively. 

## 2. Quick Start🚀

1. File Structure

```
.\data [datasets]  
.\AspectExtration  [code for extracting asepct from datasets]  
.\src  
```

1. extract aspects

```sh
python AspectExtraction/1_aspext_extractor.py
python AspectExtraction/2_aspect_merge.py
python AspectExtraction/3_aspect_filter.py
python AspectExtraction/4_aspect_sentiment.py
```

> [Note!] We use seta to calculate the review sentiment scores.  
> The package of seta need to used at root user. 



3. training the model

```sh
python src/data_process.py
python src/data_gp.py
python src/train.py
```



