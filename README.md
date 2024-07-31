# LightGCN++
Source code dataset for **"Revisiting LightGCN: Unexpected Inflexibility, Inconsistency, and A Remedy Towards Improved Recommendation"** (RecSys 2024 Short Paper). 

---

A supplementary document is available [here](supplementary_document.pdf).

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

## Acknowledgement
This code is implemented based on the open source [LightGCN PyTorch code](https://github.com/gusye1234/LightGCN-PyTorch).
