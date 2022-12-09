# ConReader

Code for our EMNLP 2022 paper,
**ConReader: Exploring Implicit Relations in Contracts for Contract Clause Extraction**

Weiwen Xu, Yang Deng, Wenqiang Lei, Wenlong Zhao, Tat-Seng Chua, Wai Lam

## Data Preparation
Please get the data from [cuad](https://github.com/TheAtticusProject/cuad). Place the data file in ``Data/full``, then using the following code to automatically extract definitions for legal terms. The annotated data is located at ``Data/3full``

`python  definition.py`


## Training
1. Train a CA model (base/large):
    ```
    bash run_CA.sh 
    bash run_CA-large.sh
   ```
    
2. Train a CD model (base/large):
    ```
    bash run_CD.sh 
    bash run_CD-large.sh
    ```
3. Our Training log:
    ```
    cat CA-base.log
    cat CA-large.log
    cat CD-base.log
    cat CD-large.log
    ```
## Citation
If you find this work useful, please star this repo and cite our paper as follows:
```angular2
@article{xu2022conreader,
  title={ConReader: Exploring Implicit Relations in Contracts for Contract Clause Extraction},
  author={Xu, Weiwen and Deng, Yang and Lei, Wenqiang and Zhao, Wenlong and Chua, Tat-Seng and Lam, Wai},
  journal={arXiv preprint arXiv:2210.08697},
  year={2022}
}
```# ConReader
