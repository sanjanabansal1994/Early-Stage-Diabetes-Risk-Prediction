# Early Stage Diabetes Risk Prediction

## Introduction

As part of the Machine Learning certificate program at the University of Toronto’s Data Sciences Institute, our team project explored which demographic and clinical features are the most significant predictors of early-stage diabetes risk among individuals. Using the Early-Stage Diabetes Risk Prediction dataset from the UC Irvine Machine Learning Repository, we have developed a business case, conducted data exploration, data cleaning, model selection, hyperparameter tuning, and generated actionable insights. This project highlights our competencies in data science and machine learning, showcasing our ability to tackle real-world challenges.

## Members

- [Anna Veremchuk](https://github.com/anneveremtchouk)
- [Dmytro Mozghovyi](https://github.com/DmytroMozghovyi)
- [Hadia Hussain](https://github.com/hahussain5)
- [Mehrdad Malek](https://github.com/mehrdadmalekmo)
- [Sanjana Garg](https://github.com/sanjanabansal1994)

## Project Overview

### Business Case

Insurance companies face challenges in assessing health risks and underwriting policies effectively. Early-stage diabetes, if undetected, can lead to severe health complications, increasing long-term claims and costs. A predictive model that accurately identifies individuals at risk of developing diabetes enables insurers to:

- Offer personalized health plans and proactive interventions
- Optimize premium pricing based on risk assessment
- Reduce long-term healthcare claims by encouraging early lifestyle changes

Our Early-Stage Diabetes Risk Prediction Model leverages machine learning and feature selection techniques to enhance the accuracy of risk assessment. By focusing on the most significant predictors, the model minimizes overfitting and improves performance metrics such as **F-measure, Precision-Recall Curve, and ROC-AUC**, ensuring robust decision-making for insurers.

### Business Impact

For insurance companies, integrating this predictive model into underwriting and policy management can:

- Improve Risk Stratification – Accurately segment policyholders based on diabetes risk.
- Personalized Premiums & Coverage – Offer dynamic pricing based on real-time health insights.
- Encourage Preventive Health Measures – Engage customers with targeted wellness programs.
- Reduce Claims & Improve Profitability – Lower long-term medical costs through early intervention strategies.
- By adopting data-driven risk prediction, insurers can enhance customer satisfaction, reduce uncertainties in policy underwriting, and drive business growth through proactive healthcare solutions.

## Data description
We are using the Early Stage Diabetes Risk Prediction dataset from the UC Irvine Machine Learning Repository. The dataset contains 16 features and 520 observations collected from direct questionnaires from the patients of Sylhet Diabetes Hospital in Sylhet, Bangladesh and approved by a doctor.

### Data Dictionary
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

### Dependencies
This project uses the following Python libraries

- NumPy : For fast matrix operations.
- pandas : For analysing and getting insights from datasets.
- matplotlib : For creating graphs and plots.
- seaborn : For enhancing the style of matplotlib plots.
- sklearn : For linear regression analysis.
- plotly : For dynamic plots

## Methodology 

### Exploratory Analysis
​The exploratory data analysis (EDA) in the "Early Stage Diabetes Risk Prediction" project required several key steps. The dataset was loaded and initial exploration was conducted to understand the structure and the contents of the data itself (number of observations and features, data types, etc). Next, descriptive statistics was performed on all features. This involved looking at both numeric and binary features. Since age was numeric, the mean age and interquartile range was calculated, the age distribution of the data was visualized using a histogram, as well as the age distribution by class (the target variable: early-stage diabetes). For the binary features, each feature was visualized using bar graphs to look at the distribution (and whether it was even or skewed within a feature) followed by looking at each feature by class to determine whether there were any class imbalances within the features. This was also presented in tabular form. The visualizations and interpretation of the exploratory data analysis can be found below: 

**Feature distribution**
![image](https://github.com/user-attachments/assets/fe0501a0-3581-40b1-8852-d99671ff1560)

**Feature distribution by Early-stage diabetes status (class)**
![image](https://github.com/user-attachments/assets/b61b4574-6126-4a45-950d-af026a5bbc78)
Note: Red corresponds to patients positive for early-stage diabetes and blue corresponds to patients negative for early-stage diabetes. 

1. **Age**: the age distribution is approximately normal, with most patients aged between 40 and 60 years. This suggests that middle-aged individuals constitute the majority of the dataset.
2. **Gender**: The dataset includes both male and female patients, with a higher frequency of males. However, when looking by class, there was higher proportion of females positive for early-stage diabetes than males.
3. **Polyuria**: This binary feature indicates whether a patient experiences excessive urination. The distribution shows a higher count of patients without polyuria compared to those with polyuria. When looking by class, more patients with the target variable
4. **Polydipsia**: Similar to polyuria, this binary feature represents excessive thirst. More patients do not exhibit polydipsia.​ However, more patients with early-stage diabetes reported polydipsia.
5. **Sudden Weight Loss:** The distribution reveals that more patients have not experienced sudden weight loss. When looking by class, a larger proportion of those with early-stage diabetes may report sudden weight loss.​
6. **Weakness:** A higher portion of patients report weakness, indicating it as a common symptom in the dataset.​ When looking by class, a larger proportion of those with early-stage diabetes may report weakness.​
7. **Polyphagia:** This feature denotes excessive hunger which appears to be evenly distributed. The distribution by class shows that more patients with early-stage diabetes may experience polyphagia.​
8. **Genital Thrush:** The majority of patients do not have genital thrush, with a smaller subset reporting the target condition.​
9. **Visual Blurring:** More patients experienced visual blurring, highlighting its potential relevance to diabetes risk.​
10. **Itching:** Itching appears to be evenly distributed.
11. **Irritability:** Fewer patients report irritability overall but a larger proportion of those with early-stage diabetes reporting irritability. 
12. **Delayed Healing:** Delayed healing appears to be evenly distributed.
13. **Partial Paresis:** While fewer patients reported partial peresis overall, the distribution shows that a higher proportion of those reporting partial paresis were positive for early-stage diabetes.
14. **Muscle Stiffness:** Muscle stiffness is reported by a smaller subset of patients.
15. **Alopecia:** The majority of patients do not have alopecia (hair loss), with a minor fraction affected with early stage diabetes.
16. **Obesity:** While fewer patients reported obesity, a larger proportion of those reporting obesity were positive for early stage diabetes.
    
These steps collectively provide a comprehensive understanding of the dataset, facilitating the development of predictive models for early-stage diabetes risk.

### Data Preprocessing 
After exploratory data analysis, several steps were taken to clean the data to prepare for modelling. This included creating a new categorical variable for age group and one hot encoding this to get dummy variables as well as recoding all binary variables from Yes/No to [0,1]. 

### Model Selection and Training 
### Model Optimization
## Results 
## Key Findings


