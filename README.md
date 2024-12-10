# LightGCN++
Source code and datasets for **"Revisiting LightGCN: Unexpected Inflexibility, Inconsistency, and A Remedy Towards Improved Recommendation"** (RecSys 2024 Short Paper). 

****Paper:**** [link](https://dl.acm.org/doi/10.1145/3640457.3688176)

****Supplementary document:**** [link](supplementary_document.pdf)


## How to Run the Code
The source code of LightGCN++ can be found [here](code).

* To run LightGCN++ with the specific configuration for each dataset, simply run:
```
./run.sh
```

* For the version supoorting Intel Gaudi devices:
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

## Acknowledgement
This code is implemented based on the open source [LightGCN PyTorch code](https://github.com/gusye1234/LightGCN-PyTorch).
This research was supported in part by the NAVER-Intel Co-Lab.
The work was conducted by KAIST and reviewed by both NAVER and Intel.
