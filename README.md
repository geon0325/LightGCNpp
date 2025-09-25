# LightGCN++
Source code and datasets for **"Revisiting LightGCN: Unexpected Inflexibility, Inconsistency, and A Remedy Towards Improved Recommendation"** (RecSys 2024 Short Paper). 

- ****Paper:**** [link](https://dl.acm.org/doi/10.1145/3640457.3688176) (Supplementary document: [link](supplementary_document.pdf))

The extended journal version is published in TORS.

- ****Paper:**** [link](https://dl.acm.org/doi/10.1145/3760763)

## How to Run the Code
The source code of LightGCN++ can be found [here](code).

* To run LightGCN++ with the specific configuration for each dataset, simply run:
```
./run.sh
```

* For the version supporting Intel Gaudi devices:
```
./run_gaudi.sh
```

* To run with different $\alpha$, $\beta$, and $\gamma$, run:
```
python main.py --dataset=[DATASET NAME] --alpha [ALPHA VALUE] --beta [BETA VALUE] --gamma [GAMMA VALUE]

e.g.,
python main.py --dataset="yelp2018" --alpha 0.6 --beta -0.1 --gamma 0.1
```
* By default, we recommend using $\alpha=0.6$, $\beta=-0.1$, and $\gamma=0.2$.


## Datasets
We used five datasets: LastFM, MovieLens, Gowalla, Yelp, and Amazon. You can find them [here](data).

## SELFRec Version
We also provide a code that runs in the [SELFRec](https://github.com/Coder-Yu/SELFRec) framework. You can find it [here](SELFRec).

* To run LightGCN++ with the specific configuration for each dataset, simply run:
```
./run.sh [DATASET]

e.g., to run lastfm with its optimal hyperparameters:
./run.sh lastfm
```

The optimal hyperparameter configurations of each dataset are as follows:
| Dataset         | LastFM | CiteULike | MovieLens-1M | Gowalla | Yelp  | Amazon-Sports | Amazon-Beauty | Amazon-Book | MovieLens-10M | Alibaba |
|-----------------|--------|-----------|--------------|---------|-------|---------------|---------------|-------------|---------------|---------|
| **α**           | 0.6    | 0.5       | 0.4          | 0.6     | 0.6   | 0.6           | 0.5           | 0.6         | 0.6           | 0.6     |
| **β**           | -0.1   | -0.1      | 0.1          | -0.1    | -0.1  | -0.1          | 0.0           | -0.1        | -0.1          | 0.0     |
| **γ**           | 0.0    | 0.4       | 0.0          | 0.2     | 0.0   | 0.0           | 0.2           | 0.2         | 0.0           | 0.1     |


## Acknowledgement
This code is implemented based on the open source [LightGCN PyTorch code](https://github.com/gusye1234/LightGCN-PyTorch).
This research was supported in part by the NAVER-Intel Co-Lab.
The work was conducted by KAIST and reviewed by both NAVER and Intel.
