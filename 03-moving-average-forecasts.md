---
title: "Moving Average Forecasts"
teaching: 30
exercises: 20
---

:::::::::::::::::::::::::::::::::::::: questions 

- Question here

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Create a stationary time-series

::::::::::::::::::::::::::::::::::::::::::::::::

## Introduction


## Create subset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
```


::::::::::::::::::::::::::::::::::::: keypoints

- note here

:::::::::::::::::::::::::::::::::::::::::::::::
