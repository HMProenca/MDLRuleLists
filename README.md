# MDL Rule Lists for prediction and data mining

The most recent version of *MDL rule lists* for classification, regression and subgroup discovery can be found on the https://github.com/HMProenca/RuleList repository  or in PyPi [https://pypi.org/project/rulelist/](https://pypi.org/project/rulelist/) and installed using:
```python
pip install rulelist
```

The new version offers several improvements in terms of algorithmic performance and theory.

**Algorithmic performance improvements**:
 * It accepts numeric and categorical explanatory variables.
 * It accepts numeric or categorical target variables.
 * It accepts single or multiple target variables, which translates into: classification; regression; multi-target classification; multi-target regression.
 * It uses a beam-search for searching the candidates which makes it several orders of magnitude faster.
 * It can perform subgroup list discovery.

**Theory improvements**:
 * Uses the Normalized Maximum Likelihood (NML) to encode categorical variables, instead of the prequential plug-in code.
 * Uses Bayesian Gaussian code for numeric targets.


## Example of usage:

```python
import pandas as pd
from rulelist import RuleList
from sklearn import datasets
from sklearn.model_selection import train_test_split

task = 'prediction'
target_model = 'categorical'

data = datasets.load_breast_cancer()
Y = pd.Series(data.target)
X = pd.DataFrame(data.data)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)

model = RuleList(task = task, target_model = target_model)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test.values,y_pred)

print(model)
```




## Contact

If there are any questions or issues, please contact me by mail at `hugo.manuel.proenca@gmail.com` or open an issue here on Github.


## Citation

In a machine learning (prediction) context for problems of classification, regression, multi-label classification, multi-category classification, or multivariate regression cite the corresponding bibtex of the first classification application of MDL rule lists:

```
@article{proencca2020interpretable,
  title={Interpretable multiclass classification by MDL-based rule lists},
  author={Proen{\c{c}}a, Hugo M and van Leeuwen, Matthijs},
  journal={Information Sciences},
  volume={512},
  pages={1372--1393},
  year={2020},
  publisher={Elsevier}
}
```

in the context of data mining and subgroup discovery please refer to subgroup lists:

```
@article{proencca2020discovering,
  title={Discovering outstanding subgroup lists for numeric targets using MDL},
  author={Proen{\c{c}}a, Hugo M and Gr{\"u}nwald, Peter and B{\"a}ck, Thomas and van Leeuwen, Matthijs},
  journal={arXiv preprint arXiv:2006.09186},
  year={2020}
} 
```

# References #
 * [Interpretable multiclass classification by MDL-based rule lists. Hugo M. Proença, Matthijs van Leeuwen. Information Sciences 512 (2020): 1372-1393.](https://www.sciencedirect.com/science/article/pii/S0020025519310138) or publicly available in [ArXiv](https://arxiv.org/abs/1905.00328) -- experiments code (old version) available [here](https://github.com/HMProenca/MDLRuleLists)
 * [Discovering outstanding subgroup lists for numeric targets using MDL. Hugo M. Proença,Thomas Bäck, Matthijs van Leeuwen. ECML-PKDD(2020): ](https://arxiv.org/abs/2006.09186) -- experiments code available [here](https://github.com/HMProenca/SSDpp-numeric)
