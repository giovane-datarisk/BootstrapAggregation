### Como Utilizar Bootstrap Aggregation Para Classificação

Aqui você encontra os notebooks utilizados no artigo "Como Utilizar Bootstrap Aggregation Para Classificação", com cada etapa desenvolvida.

**Bibliotecas Utilizadas**

```Python
import pandas as pd
import numpy as np
from numpy import arange
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
import seaborn as sns
```

Para outros conteúdos acesse https://medium.com/datarisk-io.
