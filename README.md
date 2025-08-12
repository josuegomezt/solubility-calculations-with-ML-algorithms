# solubility-calculations-with-ML-algorithms
Prediction of logS Values for Different Molecules Using Machine Learning Algorithms, and Enhancement of Various Solubility Prediction Models Through Ensemble Learning Methods, Quantum-Chemical Molecular Descriptors, and Data Augmentation Techniques to Improve the Feature Space

## Directory Descriptions

- **aqsolpred/**: Directory for the aqsolpred model and its modules (Sorkun et al., iScience 24, 101961, January 22, 2021. DOI: 10.1016/j.isci.2020.101961)  
  - Contains the implementation of the aqsolpred model and associated modules for solubility prediction.

- **initial_benchmarking.ipynb**: Contains the initial benchmarking of the project, where 42 ML regressors and an MLP regressor are trained to compare their performance. Insights are proposed based on various sources.

![Copy of = NC](https://github.com/user-attachments/assets/4e1977c5-9b8b-41a3-ab6c-1036a47cc86d)

In this project, the solubility of organic molecules (as $logS$) is predicted using various Machine Learning algorithms and Neural Networks, starting from a set of molecular physicochemical descriptors (obtained with the RDKit Python library) as the feature space and solubility data as the targets. The performance of these models is compared with experimental values obtained in the laboratory to evaluate which model is more efficient at predicting the solubility of the molecules in an aqueous medium.

Finally, **methodologies for improving the results are proposed.** ***One approach is based on the ensemble model*** suggested by Sorkun et al. (iScience 24, 101961, January 22, 2021. DOI: 10.1016/j.isci.2020.101961), ***where solubility prediction is enhanced by averaging the results from three Machine Learning models (Neural Networks, XGBoost, and Random Forest)***. In this approach, the physicochemical descriptors are selected through Lasso regression to use only those relevant to the prediction task, thereby refining the feature space and improving the model's accuracy. ***Another approach is inspired by*** the work of Yao et al. (Journal of Materials Chemistry A 10(30), January 2022. DOI: 10.1039/D2TA03728A), which proposes ***calculating the electrostatic potentials on the Van der Waals surface of the training compounds. These electrostatic potentials are then provided as features to the Machine Learning algorithms.*** By offering a more accurate description of the interactions between the molecules of a given compound, this method helps improve the model’s performance in solubility prediction.

An **additional strategy implemented** in this project is the use of Generative Adversarial Networks, specifically a WGAN with Gradient Penalty (WGAN-GP), for data augmentation. This approach expands the available chemical space for training the predictive models, thus enhancing their predictive power. When working with this approach—or, in general, with any Machine Learning technique involving molecules—it is essential to handle the **structural information** of molecules, i.e., the spatial arrangement of atoms and electrons that gives them their distinct properties. To address this, different molecular input representations are explored, such as **SELFIES\***, which, unlike other molecular encoding systems, are better suited for Machine Learning models because they allow random modifications of molecular strings without the risk of generating invalid chemical structures (Krenn et al., *Machine Learning: Science and Technology* 1, 045024, 2020. DOI: 10.48550/arXiv.1905.13741).

All these approaches are applied with the goal of improving solubility prediction—an essential physicochemical property of electroactive molecules that are candidates for the design of **redox flow batteries (RFBs)**, the main focus of our research group.

***References:***

* Sorkun et al., iScience 24, 101961, January 22, 2021. DOI: 10.1016/j.isci.2020.101961  
* Yao et al., Journal of Materials Chemistry A 10(30), January 2022. DOI: 10.1039/D2TA03728A  
* Li, Y., et al. (2020). Improved Prediction of Aqueous Solubility of Novel Compounds by Going Deeper With Deep Learning. Frontiers in Oncology, 10, 121. DOI: 10.3389/fonc.2020.00121  
* Krenn et al., *Machine Learning: Science and Technology* 1, 045024, 2020. DOI: 10.48550/arXiv.1905.13741  

