---
title: Setup
---

The source dataset this used for lesson consists of smart
meter power consumption data provided by the 
Los Alamos Public Utility Department (LADPU) in Los Alamos, New Mexico,
USA. In their original format the data have only been 
processed to remove consumer information. The data contain
missing and duplicate values.

The original dataset on which the lesson materials are based
is available from Dyrad, _LADPU Smart Meter Data_,
[https://doi.org/10.5061/dryad.m0cfxpp2c](https://doi.org/10.5061/dryad.m0cfxpp2c)
and has been made available with a CC-0 license:
 
>Souza, Vinicius; Estrada, Trilce; Bashir, Adnan; Mueen, Abdullah (2020), 
>_LADPU Smart Meter Data_, Dryad, Dataset, 
>[https://doi.org/10.5061/dryad.m0cfxpp2c](https://doi.org/10.5061/dryad.m0cfxpp2c)


## Data Sets

For this lesson, the data have been modified to support the lesson objectives
without requiring a download of the full source dataset from Dryad. Because the source
data are large and require cleaning, additional steps have been taken to generate
a subset ready for use in this lesson. These steps include:

- Excluding data from meters that were not participating for the full period 
between January 1, 2014 and December 31, 2019.
- Excluding data from meters that have missing or duplicate readings, or other
anomalies.
- Further limiting the included date ranges to exclude common outliers across 
datasets due to weather events, power outages, or other causes.
- Selection of the final set of 15 data files based on inspection of plots
and completeness of the data.

At the outset of a lesson, learners are recommended to create a project directory.

1. **Download** this data file to your computer: [Smart meter data subset](https://digitalrepository.unm.edu/context/library_data/article/1003/type/native/viewcontent)
1. Within a directory on their system for which learners have read and write 
permissions (user home, desktop, or similar), create a directory named
*pandas_timeseries*.
1. In the *pandas_timeseries* directory, create a subdirectory named
*data.* Unzip the downloaded data into this directory.
1. In the *pandas_timeseries* directory, create two more directories,
*scripts* and *figures*.

Throughout the lesson, we will be creating scripts in the *scripts* directory.
If using Jupyter Notebooks, be sure to navigate to this directory before
creating new notebooks!


## Software Setup

::::::::::::::::::::::::::::::::::::::: discussion

### Details

The lesson is written in Python. We recommend the Anaconda Python distribution,
which is available for all operating systems and comes with most of the
necessary libraries installed. Information on how to download and install
for different operating systems is available from the 
[Anaconda](https://www.anaconda.com/download) website.

There are different options for running a Python environment.

:::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::: solution

### Using Jupyter Notebook

:::::::::::::::::::::::::

:::::::::::::::: solution

### Using your preferred IDE

- Spyder
- PyCharm

:::::::::::::::::::::::::


:::::::::::::::: solution

### Using a command line client

1. Open a shell and enter ```python3```

:::::::::::::::::::::::::

