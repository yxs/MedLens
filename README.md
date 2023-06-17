# MedLens: Improve mortality prediction via medical signs selecting and regression interpolation

## Introduction

Monitoring the health status of patients and predicting mortality in advance is vital for providing patients with timely care and treatment. Massive medical signs in Electronic Health Records (EHR) are fitted into advanced machine learning models to make predictions. However, the data-quality problem of original clinical signs is less discussed in the literature. Based on an in-depth measurement of the missing rate and correlation score across various medical signs and a large amount of patient hospital admission records, we discovered the comprehensive missing rate is extremely high, and a large number of useless signs could hurt the performance of prediction models. Then we concluded that only improving data-quality could improve the baseline accuracy of different prediction algorithms. We designed MEDLENS, with an automatic vital medical signs selection approach via statistics and a flexible interpolation approach for high missing rate time series. After augmenting the data-quality of original medical signs, MEDLENS applies ensemble classifiers to boost the accuracy and reduce the computation overhead at the same time. It achieves a very high accuracy performance of 0.96 AUC-ROC and 0.81 AUC- PR, which exceeds the previous benchmark.

## Citation

If you use this code in your research, please cite the following publication.

```
@article{ye2023medlens,
  title={MedLens: Improve mortality prediction via medical signs selecting and regression interpolation},
  author={Ye, Xuesong and Wu, Jun and Mou, Chengjie and Dai, Weinan},
  journal={arXiv preprint arXiv:2305.11742},
  year={2023}
}
```

Please be sure also to cite the original [MedLens Paper]()

## Prerequsites

- Python 3.x
- numpy
- pandas
- sklearn