# LightGCN++
Source code dataset for **"Revisiting LightGCN: Unexpected Inflexibility, Inconsistency, and A Remedy Towards Improved Recommendation"** under submission. 

---

A supplementary document is available [here](supplementary_document.pdf).

---

Graph Neural Networks (GNNs) have emerged as effective tools in recommender systems. Among a variety of GNN models, LightGCN is distinguished by its simplicity and outstanding performance. Its efficiency has led to widespread adoption in a range of recommender systems across various domains including social, bundle, and multimedia recommendations. In this paper, we thoroughly examine the operational mechanisms of LightGCN, focusing on its strategies for scaling embeddings, aggregating neighbors, and pooling embeddings across layers. Our analysis reveals that, contrary to our expectation based on its formulation, LightGCN suffers from inflexibility and inconsistency when applied to real-world data.

We introduce LightGCN++, an enhanced version designed to substantially improve the recommendation quality of LightGCN by addressing its empirical inflexibility and inconsistencies. LightGCN++ incorporates flexible scaling of embedding norms and neighbor weighting, along with a tailored approach for pooling layer-wise embeddings to resolve the identified inconsistencies. Despite its remarkably simple remedy, our extensive experimental results demonstrate that LightGCN++ significantly outperforms LightGCN in recommendation performance, achieving an improvement of up to 17.81% in terms of NDCG@20. Furthermore, state-of-the-art models that utilize LightGCN as their backbone for item, bundle, multimedia, and knowledge-graph-based recommendation, exhibit improved performance when equipped with LightGCN++.

---

## How to Run the Code
* To run LightGCN++ with the specific configuration for each dataset, simply run:
```
./run.sh
```
* To run with different $\alpha$, $\beta$, and $\gamma$, run:
```
python main.py --dataset=[DATASET NAME] --alpha [ALPHA VALUE] --beta [BETA VALUE] --gamma [GAMMA VALUE]

e.g.,
python main.py --dataset="yelp2018" --alpha 0.6 --beta -0.1 --gamma 0.1
```

## Datasets
We used five datasets: LastFM, MovieLens, Gowalla, Yelp, and Amazon. You can find them [here](data).
