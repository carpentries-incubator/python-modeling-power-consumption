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
import matplotlib.pyplot as plt
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

The dataset is large, with multiple types of seasonality occurring
including

- daily
- seasonal
- yearly

trends. Additionally, the data represent smart meter readings taken from a
single meter every fifteen minutes over the course of three years. This gives
us a dataset that consists of 105,012 rows of meter readings taken at a 
frequency which makes baseline forecasts less effective.

![Plot of readings from a single meter, 2017-2019](./fig/ep2_fig1_plot_all.png)

For our current purposes, we will subset the data to a period with fewer
seasonal trends. Using datetime indexing we can select a subset of the data
from the first six months of 2019.

```python
jan_june_2019 = df["2019-03": "2019-07"].copy()
```

We will also resample the data to a weekly frequency.

```python
weekly_usage = pd.DataFrame(jan_june_2019.resample("W")["INTERVAL_READ"].sum())
print(weekly_usage.info()) # note the index range and freq
print(weekly_usage.head())
```
```output
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 23 entries, 2019-03-03 to 2019-08-04
Freq: W-SUN
Data columns (total 1 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   INTERVAL_READ  23 non-null     float64
dtypes: float64(1)
memory usage: 368.0 bytes
None
               INTERVAL_READ
INTERVAL_TIME               
2019-03-03           59.1300
2019-03-10          133.3134
2019-03-17          118.9374
2019-03-24          120.7536
2019-03-31           88.9320
```

Plotting the total weekly power consumption with a 4 week rolling mean
shows that there is still an overall trend and some apparent weekly seasonal
effects in the data. We will see how these different features of the data
influence different baseline forecasts.

```python
fig, ax = plt.subplots()

ax.plot(weekly_usage["INTERVAL_READ"], label="Weekly")
ax.plot(weekly_usage["INTERVAL_READ"].rolling(window=4).mean(), label="4 week average")

ax.set_xlabel('Date')
ax.set_ylabel('Power consumption')
ax.legend(loc=2)

fig.autofmt_xdate()
plt.tight_layout()
```

![Plot of weekly readings from a single meter, 2019](./fig/ep2_fig2_plot_weekly.png)


::::::::::::::::::::::::::::::::::::: keypoints

- Use *test* and *train* datasets to evaluate the performance of different
models.
- Use *mean average percentage error* to measure a model's performance.

:::::::::::::::::::::::::::::::::::::::::::::::
