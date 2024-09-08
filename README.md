# solubility-calculations-with-ML-algorithms
Prediction of logS Values for Different Molecules Using Machine Learning Algorithms, and Enhancement of Various Solubility Prediction Models Through Ensemble Learning Methods and Quantum-Chemical Molecular Descriptors That Improve the Feature Space


## Directory Descriptions

- **aqsolpred/**: Directory for the aqsolpred model and its modules (Sorkun et al., iScience 24, 101961, January 22, 2021. DOI: 10.1016/j.isci.2020.101961)
  - Contains the implementation of the aqsolpred model and associated modules for solubility prediction.

- **initial_benchmarking.ipynb**: Contains the initial benchmarking of the project, where 42 ML regressors and an MLP regressor are trained to compare their performance. Insights are proposed based on various sources.

![Copy of = NC](https://github.com/user-attachments/assets/4e1977c5-9b8b-41a3-ab6c-1036a47cc86d)

In this project, the solubility of organic molecules (as $logS$) is predicted using various Machine Learning algorithms and Neural Networks, starting from a set of molecular physicochemical descriptors (obtained with RDKit python library) as the feature space and solubility data as the targets. The performance of these models is compared with experimental values obtained in the laboratory to evaluate which model is more efficient at predicting the solubility of the molecules in an aqueous medium.

Finally, **methodologies for improving the results are proposed.** ***One approach is based on the ensemble model*** suggested by Sorkun et al. (iScience 24, 101961, January 22, 2021. DOI: 10.1016/j.isci.2020.101961), ***where solubility prediction is enhanced by averaging the results from three Machine Learning models (Neural Networks, XGBoost, and Random Forest)***. In this approach, the physicochemical descriptors are selected through Lasso regression to use only those relevant to the prediction task, thereby refining the feature space and improving the model's accuracy. ***Another approach is inspired by***  the work of Yao et al. (Journal of Materials Chemistry A 10(30), January 2022. DOI: 10.1039/D2TA03728A), which proposes ***calculating the electrostatic potentials on the Van der Waals surface of the training compounds. These electrostatic potentials are then provided as features to the Machine Learning algorithms.*** By offering a more accurate description of the interactions between the molecules of a given compound, this method helps improve the model’s performance in solubility prediction.

In this project, the goal is to combine the two approaches mentioned in these papers, with the hope of achieving even better results. The project is ongoing; currently, this is the pipeline being followed and the insights gained so far. The utility of this project lies in accelerating the discovery of electroactive materials for the design of redox flow batteries (RFBs) focused on clean energy.

***References:***

* Sorkun et al., iScience 24, 101961, January 22, 2021. DOI: 10.1016/j.isci.2020.101961
* Yao et al., Journal of Materials Chemistry A 10(30), January 2022. DOI: 10.1039/D2TA03728A
* Li, Y., et al. (2020). Improved Prediction of Aqueous Solubility of Novel Compounds by Going Deeper With Deep Learning. Frontiers in Oncology, 10, 121. DOI: 10.3389/fonc.2020.00121
