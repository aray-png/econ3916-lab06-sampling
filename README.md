# Sampling for Machine Learning — Train-Test Splits

## Objective

I explored how train-test split design affects machine learning evaluation and demonstrated why preventing data leakage and using representative samples matter more than simply having a large sample.

## Methodology

- Loaded the Titanic dataset, containing 891 passengers, and examined survival rates overall and by passenger class.
- Prepared the features `Pclass`, `Age`, `SibSp`, `Parch`, and `Fare`, removing rows with missing values for the initial split comparison. This left 714 passengers.
- Compared a naive 80/20 train-test split with a stratified 80/20 split using survival as the stratification target.
- Compared passenger-class shares across the overall data and the training and test sets.
- Demonstrated data leakage by fitting a `StandardScaler` on the full dataset before splitting, then corrected the process by splitting first and fitting the scaler only on the training data.
- Demonstrated imputation leakage by filling missing ages using the mean from all 891 rows, then corrected it by fitting a `SimpleImputer` on the training rows and applying it to the test rows.
- Wrote and tested `split_by_share()` to split a DataFrame using a specified test share and random seed, without stratification.
- Simulated a Literary Digest-style sampling failure using 10 million simulated voters, comparing an income-biased sample with a simple random sample of 50,000 voters.
- Calculated Meng's data defect correlation and compared the sampling errors across 100 replications.

## Key Findings

- In the original Titanic dataset, 342 of 891 passengers survived, for an overall survival rate of 0.384. Survival rates differed by passenger class: 0.630 in first class, 0.473 in second class, and 0.242 in third class.
- After removing rows with missing values, the clean dataset contained 714 passengers and had a survival rate of 0.406.
- The naive split produced a training survival rate of 0.410 and a test survival rate of 0.392. The stratified split produced rates of 0.406 in training and 0.406 in testing, much closer to the overall rate.
- Stratifying on survival kept the survival proportions similar, but did not directly guarantee the passenger-class proportions. Stratifying on both passenger class and survival kept both distributions close to the overall data.
- Fitting a scaler or imputer before splitting allows information from the test set to influence preprocessing. The correct approach is to split first, fit preprocessing steps on the training data only, and then transform the test data using those fitted steps.
- The `split_by_share()` function passed the lab's checks. With a 30% test share, it produced 499 training rows and 215 test rows. The training survival rate was 0.403, and the test survival rate was 0.414.
- In the Literary Digest-style simulation, the biased sample contained 4,382,503 responses and estimated the Landon vote share as 0.636, an error of +0.083. The simple random sample of 50,000 estimated the share as 0.550, an error of -0.002.
- Meng's data defect correlation was 0.1475, and the predicted sampling error was 0.0830, matching the simulated error to four decimal places.
- Across 100 replications, the biased sample had a much smaller spread of errors but remained consistently biased. The random sample's estimates varied more but stayed centered much closer to the true population share.

## Conclusion

This lab showed me that the way data is sampled and split affects how much I can trust an estimate or model evaluation. Stratification can preserve important class proportions, and fitting preprocessing steps only on training data prevents test-set information from leaking into the training process. The sampling simulation also showed that a very large biased sample can be less accurate than a much smaller random sample. Overall, sample design matters more than sample size alone.
