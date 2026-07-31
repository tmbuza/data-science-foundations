# Insights Report: Iris Dataset

## Dataset and group balance

- The dataset contains 150 observations and 8 columns.
- It contains 3 species groups.
- Group sizes range from 50 to 50 observations.

## Petal length by species

| species | n | mean | median | sd | iqr |
| --- | --- | --- | --- | --- | --- |
| setosa | 50 | 1.462 | 1.500 | 0.174 | 0.175 |
| versicolor | 50 | 4.260 | 4.350 | 0.470 | 0.600 |
| virginica | 50 | 5.552 | 5.550 | 0.552 | 0.775 |

## Strongest descriptive differentiation

`petal_length` has the largest eta-squared value
(0.941) among the summarized features.

## Interpretation

Petal measurements provide clearer descriptive differentiation among species
than sepal measurements in this dataset.

## Limitations

These results describe this dataset. They do not establish causation or
classification performance. Correlations involving `petal_area` partly reflect
how that derived feature was calculated.
