# Early Stage Diabetes Risk Prediction

## Introduction

As part of the Machine Learning certificate program at the University of Toronto’s Data Sciences Institute, our team project explored which demographic and clinical features are the most significant predictors of early-stage diabetes risk among individuals. Using the Early-Stage Diabetes Risk Prediction dataset from the UC Irvine Machine Learning Repository, we have developed a business case, conducted data exploration, data cleaning, model selection, hyperparameter tuning, and generated actionable insights. This project highlights our competencies in data science and machine learning, showcasing our ability to tackle real-world challenges.

## Members

- [Anna Veremchuk](https://github.com/anneveremtchouk)
- [Dmytro Mozghovyi](https://github.com/DmytroMozghovyi)
- [Hadia Hussain](https://github.com/hahussain5)
- [Mehrdad Malek](https://github.com/mehrdadmalekmo)
- [Sanjana Garg](https://github.com/sanjanabansal1994)

## Business Case

Insurance companies face challenges in assessing health risks and underwriting policies effectively. Early-stage diabetes, if undetected, can lead to severe health complications, increasing long-term claims and costs. A predictive model that accurately identifies individuals at risk of developing diabetes enables insurers to:

- Offer personalized health plans and proactive interventions
- Optimize premium pricing based on risk assessment
- Reduce long-term healthcare claims by encouraging early lifestyle changes

Our Early-Stage Diabetes Risk Prediction Model leverages machine learning and feature selection techniques to enhance the accuracy of risk assessment. By focusing on the most significant predictors, the model minimizes overfitting and improves performance metrics such as **F-measure, Precision-Recall Curve, and ROC-AUC**, ensuring robust decision-making for insurers.

## Business Impact

For insurance companies, integrating this predictive model into underwriting and policy management can:

- Improve Risk Stratification – Accurately segment policyholders based on diabetes risk.
- Personalized Premiums & Coverage – Offer dynamic pricing based on real-time health insights.
- Encourage Preventive Health Measures – Engage customers with targeted wellness programs.
- Reduce Claims & Improve Profitability – Lower long-term medical costs through early intervention strategies.
- By adopting data-driven risk prediction, insurers can enhance customer satisfaction, reduce uncertainties in policy underwriting, and drive business growth through proactive healthcare solutions.

## Project Overview

## Dependencies
This project uses the following Python libraries

- NumPy : For fast matrix operations.
- pandas : For analysing and getting insights from datasets.
- matplotlib : For creating graphs and plots.
- seaborn : For enhancing the style of matplotlib plots.
- sklearn : For linear regression analysis.
- plotly : For dynamic plots

## Exploratory Analysis

### Schema
|Variable Name|Type|Description |Value|
|-------------|----|------------|-----|
|Age          |int64|Age of the person|20-65|		
|Sex | object| Sex of the person|1. Male, 2.Female|	
|Polyuria| object|Excessive Urine production| 1.Yes, 2.No.|	
|Polydipsia | object|Excessive thirst|1.Yes, 2.No.|		
|Sudden weight loss| object|Sudden weight loss|1.Yes, 2.No.|
|Weakness|object|Whether the patient experienced weakness| 1.Yes, 2.No.|
|Polyphagia |object|Excessive hunger|1.Yes, 2.No.	|
|Genital thrush|object|Yeast infection that affects men and women| 1.Yes, 2.No.	|
|Visual blurring |object|Blurred vision| 1.Yes, 2.No. |	
|Itching |object|Irritating sensation on skin| 1.Yes, 2.No. |
|Irritability|object| heightened sensitivity, a tendency to become easily annoyed or angered| 1.Yes, 2.No.	|
|Delayed healing |object|Slow-healing cuts, bruises, and sores| 1.Yes, 2.No.|
|partial paresis |object|partial or mild paralysis| 1.Yes, 2.No.|		
|muscle stiffness |object|pain or tightness in muscles| 1.Yes, 2.No.|
|Alopecia |object|Hair loss| 1.Yes, 2.No.|
|Obesity |object|Excessive body fat|1.Yes, 2.No.|
|Class |object|Target variable: whether the patient is positive for early stage diabetes| 1.Positive, 2.Negative.|




