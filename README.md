# DPFuzz
DPFuzz: Fuzzing and Debugging for Differential Performance Bugs in Machine Learning Libraries

DPFuzz is a tool for fuzzing and debugging differential performance bugs. The paper and its overview are included in this folder. Please see the 'DPFuzz-Overview.pdf' to have a overall picture of DPFuzz.

The tool implemented as a part of ISSTA'20 paper: [Detecting and Understanding Real-World Differential Performance Bugs in Machine Learning Libraries](https://arxiv.org/abs/2006.01991). Please find the complete artifacts and case studies in [Virtual Box Package](https://drive.google.com/open?id=18ZxxyeyaxBZg2x15O4lT5DHKetDtI2P1).

### System Requirement
Python 2.7 with following library: scikit-learn 0.20.3, subprocess32, numpy, argparse; Rscript (util).

## Fuzzing for insertionX sort
Let us overview the fuzzing aspect of tool using an example of insertionX sort algorithm.

```bash
cd Experiments/
```

Here, let's focus on insertionX experiments. Others follow the same idea. To fuzz with DPFuzz:

```bash
python driver.py --name insertionX --size 30 --clusters 3 --max_iter 1000 > sample_outcomes/job-output-insertionX-sort-1.out
```

where we set the number of clusters to 3, the maximum size of array to 30, and maximum number of iteration of fuzzer to 1000 (under 90 mins timeout). You can find fuzzing for other benchmarks and also with
other fuzzers (SlowFuzz and PerfFuzz) in the virtual image package.

Now, let us overview how we can use our approach to detect and explain performance
bugs in machine learning libraries.


## Case Study with Logistic Regression

First, let's cd to Case-Studies folder:

```bash
cd Case-Studies/
ls
```

As you can see, there are five folders here.
They correspond to the five steps described in the 'DPFuzz-Overview.pdf' file.

### Fuzzing

Let's start with fuzzing for Logistic Regression (this corresponds to the overview Section):

```bash
cd DPFuzz
python driver.py --name LogisticRegression --size 100000 --clusters 4 --max_iter 200 --num_param 14 --size_index 0 > job-outputs/job-output-LogisticRegression-1.txt
```

here, we run fuzzer for 144 minutes (or a maximum of 200 iterations) where the maximum input size, clusters, library parameters are 100000, 4, and 14.

The size index parameter indicates which parameter is size. In this case, it is the first (zero-index) parameter.

Please see 'LogisticRegression.py' as the driver for Logistic Regression and 'logistic\_regression\_Params.xml' as the input parameter file. The xml file shows the number of parameters (the number of entities) and the index of size parameter. The sample driver and input parameter files can be writing using the [API descripiton of scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

Note that a user can introduce new case studies with having a driver and the xml description of input parameters (see the samples for the case studies in this paper). While the arguments of fuzzer such as the number of clusters play an important role in the outcomes, they are not strict and we believe that users can fuzz the target application with default parameters such as the one provided with the logistic regression.

Next, we re-run the inputs on a more precise machine (such as NUC machine) in isolation to have precise time measurements. The run varies size parameters and measure the response time for a path on multiple different sizes.
The output of this step is the inputs for clustering.

### Clustering
Let's move to the second step that is clustering:

```bash
cd Case-Studies/Clustering/
```

The following command clusters for Logistic Regression:

```bash
Rscript functional_clustering.R Examples/LogisticRegression_final_time.csv 4 3 1 1
```

where the first parameter is the number of clusters, the second parameter is a noise tolerance parameters, the third parameter shows to produce label outputs, and the fourth parameter indicates to use the slope for clustering. Please see 'Commands.txt' for more information.

The issue of clustering commands will generate files inside 'Figures' and 'labels' folder.

### Classification in input space
Let's apply the decision tree classifier in the space of library input parameters to explain different performances with these features.

```bash
cd Case-Studies/Classification-input-space/
```

The inputs to the classification are provided inside 'Examples' folder. The csv files are produced using labels from the clustering and parsing the fuzzing outcomes (inputs processed on NUC machine) to csv format.
For example, the following command parses the job file for logistic regression before running it on the NUC machine:

```bash
python parse_job_output_to_csv.py ../DPFuzz/job-outputs/job-output-LogisticRegression-1.txt job-output-LogisticRegression.csv ../DPFuzz/logistic_regression_Params.xml 14
cat job-output-LogisticRegression.csv
```

Now, let's apply decision tree inference to explain different clusters based on the parameters of logistic regression:

```bash
python Classify.py --filename Examples/Logistic_Regression/LogisticRegression_input_space_labeled.csv  --output Examples/Logistic_Regression/Logistic_Regression
```

here, the decision tree classifier takes the file name as input and produce the decision tree model as outputs. The code generates 5 decision tree models, let's look at the first one:

```bash
dot -Tpng Examples/Logistic_Regression/Logistic_Regression_tree0.dot -o Examples/Logistic_Regression/tree_0.png
```

Now, you can visit the folder and open the png file. Note that a sample of decision tree outcome is provided, see 'tree\_sample.png'.

The following description shows how to interpret the decision tree model for logistic regression:

```bash
cat Examples/Logistic_Regression/Description.txt
```

As we can see in the decision tree, there is an unexpected performance differences for solver = 'newton-cg' when tolerance parameter is zero versus when it is very close to zero (say 0.00001).

We turn into program internals and obtain internal features via instrumentations and apply the decision tree inference on these features.

### Instrumentations
Let's jump into the instrumentations. Note that the instrumentation is a naive
approach to gather interesting features. The next update will provide a better
implementation for the instrumentation. Currently, you need Java 8 for the
last command in the instrumentation.

```bash
cd Case-Studies/Instrumentations/
```

Let's perform instrumentations for logistic regression:

```bash
cd Logistic_Regression/
python instrumentations.py ../inputs/job-output-LogisticRegression-1.txt
```

The output of logistic regression inside 'outputs' folder: 'logistic\_regression\_outputs\_instrumentations.txt'
The following command convert the outputs of instrumentations to csv format:

```bash
cd ../outputs
java -cp callRecord-0.1.jar stac.discriminer.parser.parserInteger logistic_regression_results_instrumentations.txt
```

Now, we are ready to apply decision tree model on the set of internal features to determine root cause of differences in Logistic Regression.

### Classification in the space of program internals
Let's move to the classification task based on library's internals:

```bash
cd Case-Studies/Classification-internal-space/
```

The inputs for the classification are provided inside 'Examples' folder. To prepare the input for logistic regression,
we combine the instrumentation csv file with the input parameter csv file (see Examples/Logistic\_Regression/LogisticRegression\_inputs\_instrumentations.csv) and filter inputs that have
solver = 'newton-cg' and label them based on whether tolerance parameter is zero or non zero (see Examples/Logistic\_Regression/LogisticRegression\_inputs\_instrumentations\_newton-cg.csv).
Finally, we remove the input parameters and use the internal features for the classification (see Examples/Logistic\_Regression/LogisticRegression\_instrumentations\_newton-cg.csv)

Now, it is time to apply decision tree inference:

```bash
python Classify.py --filename Examples/Logistic_Regression/LogisticRegression_instrumentations_newton-cg.csv --output Examples/Logistic_Regression/LogisticRegression_tree
```

Once again, this will generate 5 decision tree models. Let's convert the dot file for one of them:

```bash
dot -Tpng Examples/Logistic_Regression/LogisticRegression_tree_tree0.dot -o Examples/Logistic_Regression/tree_0.png
```

A sample decision tree is provided inside the Example folder.

Note that this is confirmed by developers as a performance bug
[logistic-regression-bug](https://github.com/scikit-learn/scikit-learn/issues/16186) and
has since fixed by them [logistic-regression-fix](https://github.com/scikit-learn/scikit-learn/pull/16266/files).
