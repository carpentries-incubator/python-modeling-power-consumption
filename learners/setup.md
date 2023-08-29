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

The Anaconda distribution recommended above includes Jupyter Notebook, which
is a browser-based electronic notebook environment that supports Python, R, 
and other languages.There are two ways that you can launch a notebook server.
The first option is to run the application from the Anaconda Navigator:

1. Launch Anaconda Navigator using your operating system's application 
launcher.
2. The Navigator is a utility for managing environments, libraries, and
applications. Find the Jupyter Notebook application and click on the *Launch*
button to start a notebook server:  ![](fig/anaconda-navigator-launch-jupyter.png){alt='Anaconda Navigator launch Jupyter'}
3. The Jupyter Notebook server will open up a file navigator in your home 
directory of your operating system. Click through to navigate to the project
*scripts* directory created in the setup, above. Click on *New* and select
*Python 3* to create a Jupyter Notebook in that directory.
![](fig/anaconda-jupyter-new-nb.png){alt='Anaconda Jupyter new Notebook'}
4. A new "Untitled" notebook will open up. When you see an empty notebook cell
you are ready to go!
![](fig/anaconda-jupyter-success.png){alt='Anaconda Jupyter success.'}


A second option is to use a command line client. 

1. Open the default command line utility for your operating system. For Mac and
many Linux systems, this will be the *Terminal* app. On Windows, it is 
recommended to launch either the *CMD.exe Prompt* or the *Powershell Prompt*
from the Navigator.
2. Use the ```cd``` or *change directory* command to navigate to the *scripts*
subdirectory of the project directory created in the setup section above.

```
cd ~/Desktop/pandas_timeseries/scripts
```

3. Launch a Jupyter Notebook server using the ```jupyter notebook``` command.
When the server launches, information similar to the below will appear in the
console:
![](fig/anaconda-jupyter-server.png){alt='Anaconda Jupyter server starting.'}

4. The Jupyter Notebook application will also open in a web browser. Click on 
*New* and select *Python 3* to create a Jupyter Notebook in that directory.
![](fig/anaconda-jupyter-new-nb.png){alt='Anaconda Jupyter new Notebook'}

5. A new "Untitled" notebook will open up. When you see an empty notebook cell
you are ready to go!
![](fig/anaconda-jupyter-success.png){alt='Anaconda Jupyter success.'}

6. When you are finished working, after closing the Jupyter browser interface,
be sure to also stop the server using ```CONTROL-C```.


:::::::::::::::::::::::::

:::::::::::::::: solution

### Using a command line client

1. Open a shell and enter ```python3```

:::::::::::::::::::::::::

