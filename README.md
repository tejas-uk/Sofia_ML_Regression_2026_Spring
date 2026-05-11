# Sofia_ML_Regression_2026_Spring

This is a homework Assignment for Sofia University Machine Learning Class.

## Description

### Overall Information
This test task requires training a regression model using a set of multivariate feature examples and correct answers from the training set. The model must then be applied to a set of feature examples from the test set, and a file containing the resulting answers must be submitted.

The features themselves do not carry any meaningful meaning.

### Winner Determination Algorithm

The initial sorting of solutions will be based on the Public test. After the competition ends, a final sorting will be performed based on the Private test. Please note: a high ranking on the Public test does not guarantee a high ranking on the Private test (so don't over-focus / overfit on the Public test)!

### Score Points

The competition includes a "Baseline" that will be worth 100 points if completed. The top three places in the competition will receive an additional 400, 300, and 250 points, respectively. Places 4-10 will receive an additional 200 points, and places 10-15 will receive an additional 100 points. To receive a score for the competition, you must provide the code for the best submission and present the techniques used.

### Submission Frequency

Up to 5 submissions per day are allowed.

### Rules

Each solution must have a name in the format "Last Name_First Name"; other solutions will not be accepted.
Teams are not allowed.
Solutions cannot be posted from different accounts.
Winners will be required to provide reproducible code and showing how their solution was obtained, and present their results during the last onground session.

### Evaluation

The quality of your solution will be assessed based on the [coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination) R2 metric, the value of which, by the way, is returned when calling the **model.score(testX, testY)** function for any regression model from the **Scikit-Learn** package.

The determination coefficient value lies within the range

$$-\infty < R^2 \leq 1$$

with the maximum value

$$R^2 = 1$$

achieved in the case of a perfect, error-free prediction.

### Submission File

For each `Id` in the test set, you must predict a probability for the `target` variable. The **csv** file should contain a header and have the following format:

```csv
Id,target
2,0.5
5,-0.09
6,1000.0
...
```