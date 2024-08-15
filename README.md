<h1 align="center">
Aspectormer
</h1>

<p align="center">
  <a href="#2-quick-start🚀">Quick Start</a> •
  <!-- <a href="https://dl.acm.org/doi/10.1145/3583780.3614770">Paper</a> • -->
  <!-- <a href="#3-citation☕️">Citation</a> -->
</p>



Repo for a under review paper of AAAI 2025: [A Graph Transformer Approach for Aspect-based Recommendation]

## 1. Introduction✨
User-generated reviews often contain certain key aspects of many items and are particularly important for understanding users' shopping interests. Recent recommendation methods build aspect graphs based on reviews and then utilize graph neural networks to embed user preferences. However, different users may choose different words or phrases to describe a certain aspect of an item, resulting in a sparse aspect graph, which hinders the information propagation between nodes in the graph. In this paper, we propose an Aspect Graph Transformer (Aspectormer) for aspect-based recommendation, which employs a Transformer to learn global structure features from aspect graphs, reducing the impact of the sparsity of the graph.
Since it is infeasible to train Transformer on large graphs, we designed an aspect filter to select a subset of aspects from the entire set for each user and item to participate in training. At the same time, the aspect filter generates a score for each aspect, which is combined with the attention mechanism in the Transformer to integrate user preferences in the learning process. Experiments on five real-world datasets demonstrate that the predictions of Aspectormer significantly outperform SOTA baselines by $2.84\%$ and $2.36\%$ in MSE and MAE, respectively. More importantly, Aspectormer can help to discover aspects of potential concern to users.

## 2. Quick Start🚀

1. File Structure

```
.\amazon_data [training data]  
.\sim_res  [save the similarity matrixs]  
.\similarity  [similarity measures]  
.\untils  [some tools to papre training data]  
.\src  
```

2. calculate user similarities and item similarities in three levels

```sh
sudo python src/get_review_sim.py  
sudo python src/get_user_sim.py
```

> [Note!] We use seta to calculate the review sentiment scores.  
> The package of seta need to used at root user. 

3. training the model

```sh
python src/main.py
```



