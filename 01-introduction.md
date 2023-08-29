---
title: "Introduction to Time-series Forecasting"
teaching: 20
exercises: 10
---

:::::::::::::::::::::::::::::::::::::: questions 

- How can we predict future values in a time-series?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Define concepts applicable to forecasting models.

::::::::::::::::::::::::::::::::::::::::::::::::

## Introduction

This lesson is the second in a series of lessons demonstrating Python
libraries and methods for time-series analysis and forecasting. 

The first lesson, 
[Time Series Analysis of Smart Meter Power Consmption Data](https://carpentries-incubator.github.io/python-pandas-power-consumption/), 
introduces datetime indexing features in the Python ```Pandas``` library. Other
topics in the lesson include grouping data, resampling by time frequency, and 
plotting rolling averages. 

This lesson introduces forecasting time-series. Specifically, this lesson aims
to progressively demonstrate the attributes and processes of the 
SARIMAX model (**S**easonal **A**uto-**R**egressive **I**ntegrated **M**oving **A**verage withe**X**ogenous factors). Exogenous factors are out of scope of
the lesson, which is structured around the process of predicting a single 
timestep of a variable based on the the historic values of that same variable.
Multi-variate forecasts are not addressed. Relevant topics include:

- Stationary and non-stationary time-series
- Auto-regression
- Seasonality

The lesson demonstrates statistical methods for testing for the presence of 
stationarity and auto-regression, and for using the ```SARIMAX``` class of the 
Python ```statsmodels``` library to make forecasts that account for these
processes. 

As noted throughout the lesson, the code used in this lesson is based on and
in some cases is a direct application of code used in the Manning Publications 
title, *Time series forecasting in Python*, by Marco Peixeiro.

> Peixeiro, Marco. Time Series Forecasting in Python. [First edition]. Manning Publications Co., 2022.

The original code from the book is made available under an 
[Apache 2.0 license](https://github.com/marcopeix/TimeSeriesForecastingInPython/blob/master/LICENSE.txt). Use and application of the code in these materials is within
the license terms, although this lesson itself is licensed under a Creative Commons
[CC-BY 4.0 license](https://creativecommons.org/licenses/by/4.0/legalcode). Any
further use or adaptation of these materials should cite the source code
developed by Peixeiro:

> Peixeiro, Marco. Timeseries Forecasting in Python [Software code]. 2022.
Accessed from [https://github.com/marcopeix/TimeSeriesForecastingInPython](https://github.com/marcopeix/TimeSeriesForecastingInPython).

The third lesson in the series is 
[Machine Learning for Timeseries Forecasting with Python](https://carpentries-incubator.github.io/python-classifying-power-consumption/).
It follows from and builds upon concepts from these first two lessons in the 
series.

All three lessons use the same data. For information about the data and how to
set up the environment so the code will work without the need to edit file paths,
see the [Setup](https://carpentries-incubator.github.io/python-modeling-power-consumption/) section.

::::::::::::::::::::::::::::::::::::: keypoints

- The Python ```statsmodels``` library includes a full featured implementation
of the SARIMAX model.

:::::::::::::::::::::::::::::::::::::::::::::::
