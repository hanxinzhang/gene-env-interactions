# Gene-environment interactions explain a substantial portion of variability of common neuropsychiatric disorders

The existence of complex interactions between genetic variation and environmental stimuli (G-by-E effects) in human disease etiology has long been suspected but measuring such interactions in the real world has proven difficult. The primary limiting factor is the availability of sufficiently large and rich data, combining genetic, environmental, and phenotypic information for the same individual. As a consequence, estimates of disease heritability have been obtained from relatively small data sets with mathematical models built on strong, simplified assumptions. The proportion of phenotypic variability interactions in most complex diseases that can be explained by G-by-E remains unknown. In this study, we dissect the etiology of ten major neuropsychiatric disorders, by analyzing data for 138 thousand US families, with nearly half a million unique individuals. We show that, while gene-environment interactions account only for a small portion of the total phenotypic variance for a subset of disorders (depression, adjustment disorder, substance abuse), they explain a rather large quantity of the remaining disorders: over 20 percent for ADHD, migraine, and anxiety/phobic disorder, and close to 30 percent for recurrent headaches, sleep disorders, and posttraumatic stress disorder. For the first time, we incorporated -- in the same analysis – clinical data, family pedigrees, the spatial distribution of individuals, their socioeconomic and demographic confounders, and a comprehensive collection of raw environmental measurements. Our work clears the path towards identifying specific interactions of genetic variants and environmental signals. If we are able to ascertain a catalog of genetic variants interacting with specific environmental stimuli, we can then design personalized environmental plans for the patient at risk.

## Data
The data sets we used are serialized in Python 3 binary pickle files (.bpkl3). The phenotypic outcomes are stored in the `sample_phe.bpkl3`:
```python
import pickle

with open('sample_phe.bpkl3', 'rb') as f:
    sample_phe_mat = pickle.load(f)['sample_phe_mat']
    
sample_phe_mat.shape

# Out = (404475, 568)
```
Note there are 404,475 unique patients in the data. For each patient, we reported if they ever had a diagnosis record for 568 diseases respectively (we only analyzed the 10 most common neuropsychiatric disorders among them).

Similarly, we have the sex and age for all the 404,475 patients:

```python
with open('sel_sample_model_variables2.bpkl3', 'rb') as f:
    var_dict = pickle.load(f)
    sex_age_fam21 = var_dict['fam21_demo']
    sex_age_fam11 = var_dict['fam11_demo']
    sex_age_fam22 = var_dict['fam22_demo']
    
sex_age_fam21.shape
# Out = (344865, 2)
```
As indicated above, there are 114,955 nuclear families consisting of 2 parents and 1 child (344,865 individuals). Similarly, we have the sex and age for families of 1 parent and 1 child (`sex_age_fam11`) and families of 2 parents and 2 children (`sex_age_fam22`). Families with more than 2 children are negligible in our data. We discarded these large families as they would complicated our computation with minimal gain.

Note the phenotype data matrix `sample_phe_mat` has been sorted by the faimly belonging, so the first 114,955 * 3 = 344,865 rows represent the families of 2 parents and 1 child. Same goes for the next rows.

We also have the EQI (environmental quality index) data for these 404,475 patients:
```python
with open('sample_eqi_design_matrix.bpkl3', 'rb') as f:
    SAMPLE_EQI = pickle.load(f)
    SAMPLE_EQI_1 = SAMPLE_EQI[:, :5]
    
SAMPLE_EQI_1.shape
# Out = (404475, 5)
```
