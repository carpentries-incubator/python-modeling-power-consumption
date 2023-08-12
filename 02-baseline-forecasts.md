---
title: "Baseline Metrics for Timeseries Forecasts"
teaching: 30
exercises: 20
---

:::::::::::::::::::::::::::::::::::::: questions 

- What are some common baseline metrics for time-series forecasting?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Identify baseline metrics for time-series forecasting.
- Evaluate performance of forecasting methods using plots and mean absolute
percentage error.

::::::::::::::::::::::::::::::::::::::::::::::::

## Introduction

In order to make reliable forecasts using time-series data, it is 
necessary to establish baseline forecasts against which to compare the
results of models that will be covered in later sections of this lesson.

In many cases, we can only predict one timestamp into the future. From the
standpoint of baseline metrics, there are multiple ways we can define a
timestep and base predictions using

- the historical mean across the dataset
- the value of the the previous timestep
- the last known value, or
- a naive seasonal baseline based upon a pairwise comparison of a set of 
previous timesteps.

## Create a data subset for basline forecasting

Rather than read a previously edited dataset, for each of the episodes in this
lesson we will read in data from one or more of the Los Alamos Department of
Public Utilities smart meter datasets downloaded in the 
[Setup]("https://carpentries-incubator.github.io/python-modeling-power-consumption/index.html")
 section. 
 
 Once the dataset has been read into memory, we will create a datetime index,
 subset, and resample the data for use in the rest of this episode.
 First, we need to import libraries.
 
 ```python
import pandas as pd
import numpy as np
 ```
 
 Then we read the data and create the datetime index.
 
```python
df = pd.read_csv("../data/ladpu_smart_meter_data_10.csv")
print(df.info())
```

```output
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 105012 entries, 0 to 105011
Data columns (total 5 columns):
 #   Column         Non-Null Count   Dtype  
---  ------         --------------   -----  
 0   INTERVAL_TIME  105012 non-null  object 
 1   METER_FID      105012 non-null  int64  
 2   START_READ     105012 non-null  float64
 3   END_READ       105012 non-null  float64
 4   INTERVAL_READ  105012 non-null  float64
dtypes: float64(3), int64(1), object(1)
memory usage: 4.0+ MB
```

```python
# Set datetime index
df.set_index(pd.to_datetime(df["INTERVAL_TIME"]), inplace=True)
df.sort_index(inplace=True)
print(df.info())
```
```output
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 105012 entries, 2017-01-01 00:00:00 to 2019-12-31 23:45:00
Data columns (total 5 columns):
 #   Column         Non-Null Count   Dtype  
---  ------         --------------   -----  
 0   INTERVAL_TIME  105012 non-null  object 
 1   METER_FID      105012 non-null  int64  
 2   START_READ     105012 non-null  float64
 3   END_READ       105012 non-null  float64
 4   INTERVAL_READ  105012 non-null  float64
dtypes: float64(3), int64(1), object(1)
memory usage: 4.8+ MB
```








::::::::::::::::::::::::::::::::::::: keypoints

- Use *test* and *train* datasets to evaluate the performance of different
models.
- Use *mean average percentage error* to measure a model's performance.

:::::::::::::::::::::::::::::::::::::::::::::::
