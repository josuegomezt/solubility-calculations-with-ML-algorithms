# solubility-calculations-with-ML-algorithms
Prediction of logS values for different molecules by using Machine Learning algorithms


## Directory Descriptions

- **aqsolpred/**: Directory for the aqsolpred model and its modules (Sorkun et al., iScience 24, 101961, January 22, 2021. DOI: 10.1016/j.isci.2020.101961)
  - Contains the implementation of the aqsolpred model and associated modules for solubility prediction.

- **initial_benchmarking.ipynb**: Contains the initial benchmarking of the project, where 42 ML regressors and an MLP regressor are trained to compare their performance. Insights are proposed based on various sources.


Project Organization
.
├── aqsolpred/                            : Directory for the aqsolpred model and its modules (Sorkun et al., iScience 24, 101961, January 22, 2021. DOI: 10.1016/j.isci.2020.101961)
└── initial_benchmarking.ipynb            : Contains the initial benchmarking of the project, training 42 ML regressors and an MLP regressor to compare performance, and proposing insights based on different sources


![Copy of = NC](https://github.com/user-attachments/assets/4e1977c5-9b8b-41a3-ab6c-1036a47cc86d)

In this project, the solubility of organic molecules is predicted using various Machine Learning algorithms and Neural Networks, starting from a set of molecular physicochemical descriptors (obtained with RDKit) as the feature space and solubility data as the targets. The performance of these models is compared with experimental values obtained in the laboratory to evaluate which model is more efficient at predicting the solubility of the molecules in an aqueous medium.
