# Dataset notes

These files came with the original class project. The obesity CSV was checked against UCI's canonical download and matches. The breast cancer feature values and labels were checked against scikit-learn's canonical copy and match.

## Obesity

- Local file: `obesity.csv` (the original `ObesityDataSet.csv`, renamed).
- Source: [Estimation of Obesity Levels Based On Eating Habits and Physical Condition](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition).
- Credit: Fabio Mendoza Palechor and Alexis De la Hoz Manotas; UCI Machine Learning Repository, 2019. [Dataset DOI](https://doi.org/10.24432/C5H31Z).
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- 2,111 rows, 16 input features, seven weight categories.
- UCI reports that 77% of the rows were generated synthetically with SMOTE. The file does not identify original versus synthetic rows, or their parent records. A random split cannot establish independence between those groups.
- Exact duplicate rows are removed before splitting. Near duplicates and synthetic relatives may remain.

## Breast cancer

- Local file: `breast_cancer.csv` (the original `data.csv`, renamed).
- Source: [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).
- Credit: William Wolberg, Olvi Mangasarian, Nick Street and W. Street; UCI Machine Learning Repository, 1993. [Dataset DOI](https://doi.org/10.24432/C5DW2B).
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- 569 rows; diagnosis is B (benign) or M (malignant). Thirty numeric measurements describe cell nuclei in images from fine needle aspirates.
- The supplied CSV uses readable feature names such as `radius_mean`. The benchmark removes the ID column from model inputs and drops any entirely empty columns.

## Local processing

Raw CSV values are retained. Renaming the files is the only change to the supplied files. Deduplication, ID removal and target mapping happen in `src/benchmark.py`. Dataset hashes and duplicate counts are saved in `reports/`.

The dataset licenses apply to the data independently of any license later chosen for the code. This repository does not claim ownership of the datasets.

## Reading the feature names

The obesity file retains the source abbreviations:

| Feature | Meaning |
| --- | --- |
| `FAVC` | Frequent consumption of high-calorie food |
| `FCVC` | Frequency of vegetable consumption |
| `NCP` | Number of main meals |
| `CAEC` | Food consumption between meals |
| `CH2O` | Water consumption |
| `SCC` | Monitoring calorie consumption |
| `FAF` | Physical activity frequency |
| `TUE` | Time using technology devices |
| `CALC` | Alcohol consumption |
| `MTRANS` | Transportation method |

The remaining inputs are gender, age, height, weight, family history of overweight and smoking. Fractional values in some questionnaire-derived numeric fields reflect the supplied synthetic data; they are not rounded into invented discrete responses.

For breast cancer, each of ten nucleus measurements has a mean, standard-error (`_se`) and worst-value (`_worst`) version. The UCI source describes the worst value as the mean of the three largest values. These are image-derived measurements rather than lifestyle questionnaire responses.
