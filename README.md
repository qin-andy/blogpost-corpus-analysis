### Linguistic Analysis of Blogposts
Andy Qin
CSE 163 AB

# Included in this file:
* The processer.py and analyzer.py modules
* 3 sets of visualizations from 3 different subsets of the data (labelled two, five, and six)
* 2 raw subsets of the data (five_thousand_test and two_thousand_test)
* 3 pre-cleaned pickles files containing the dataframes post processing
(five_cleaned, six_cleaned, and two_cleaned)

Dataset courtesy of http://u.cs.biu.ac.il/~kop pel/BlogCorpus.htm

# Installing the correct libraries
These modules rely on the external libraries lxml and spacy in addition to the cse163 environment.
In the conda terminal, run the follwing commands:

 - ``conda install -c conda-forge spacy``
 - ``conda install -c conda-forge spacy-lookups-data``
 - ``python -m spacy download en_core_web_sm``
 - ``conda install -c anaconda lxml``

Visit https://lxml.de/installation.html and https://spacy.io/usage for more other python managers.

# Processing and Analyzing two_hundred_test
Processing this file should take about 30 seconds.
The code is already set up to process and analyze the two_hundred_test subset. Please note
that the processor may take a while to process.
1. Open processer.py
2. Ensure that the directory variable in the main method is the local path to the desired dataset.
In this case, that is 'data\\two_hundred_test\\'.
3. Run the program. The console should display different timings for each file processed.
4. When it is complete, it will export the dataframe as 'cleaned_data.pickle' for use with
the analyzer module.
5. Open the analyzer module.
6. Ensure that the directory variable in the main method paths to 'cleaned_data.pickle'.
7. Run the main method. This should print the dataframe to the console and then the
machine learning model's training accuracy and test accuracy. Note that due to
the two_hundred_test dataset being extremely small, the analyses and information are not
completely accurate. The model's accuracy may be extremely high.
8. You may view the generated visualization png files in the same directory the analyzer
is run. They should be extremely similar if not identical to the matching set of
png files in the included 'visualizations' folder, corresponding to the images prefixed by 
'two'.

# Processing and Analyzing five_thousand_test
Note that five_thousand_test may take substantially longer.
* To analyze a different dataset, reopen the processer module and change the directory variable
to the local path of five_thousand_test, which is 'data\\five_thousand_test\\'
* Repeat the steps above. The analyzer module may take longer to generate the images, but they
should correspond to their respective visualizations in the visualizations folder. The
machine learning model should have an accuracy in the 68-72% range.

# Analyzing six_thousand_test
* Note that the six_thousand_test blog xml files are not included. They can be found at
the link in the project report, if desired.
* However, the processed pickle file containing the data for six_thousand_test is included
in the 'cleaned pickles' folder. This can be used in the analyzer module for analysis.
* Make a copy of the 'six_cleaned.pickle' file and move it to the same directory as the
analyzer module. Remove any previous pickle files and rename 'six_cleaned.pickle' to
'cleaned_data.pickle'
* Run the analyzer module.
* You may view the generated visualization png files in the same directory the analyzer
is run. They should be extremely similar if not identical to the matching set of
png files in the included 'visualizations' folder, corresponding to the images prefixed by 
'six'.
