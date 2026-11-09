# Analysis of Airbnb Listings Across New York and Boston

## Problem

How do Airbnb listing prices and room types differ between New York (NY) and
Massachusetts / Boston (MA), and what listing characteristics are associated
with price? This project compares 3-bedroom listing prices between the two
states, models price from listing attributes, and classifies and clusters
listings by room type and price.

## Data

The analysis uses an Airbnb listings dataset with (at least) these columns:
`id`, `host_id`, `street`, `neighbourhood_cleansed`, `city`, `state`,
`zipcode`, `latitude`, `longitude`, `property_type`, `room_type`,
`accommodates`, `bathrooms`, `bedrooms`, `beds`, `bed_type`,
`review_scores_rating`, and `price`.

**The data file is not included in this repository.** To run the code:

1. Obtain the listings CSV from its original source (the Inside Airbnb
   listings data for New York City and Boston, <https://insideairbnb.com/>,
   combined into a single file with a `state` column distinguishing `NY`
   and `MA` rows, is the dataset this coursework was built on).
2. Create a `data/` folder in the repository root.
3. Save the combined file as `data/airbnb_listings.csv`.

The script loads exactly that relative path — there is no interactive file
picker.

## Methods

All analysis is in `Airbnb_project_code.r`:

1. **Hypothesis testing** — outlier removal and normality checks on
   3-bedroom prices, then two-sample and one-sample t-tests on seeded
   random samples of 30 listings per state. Results should be read through
   the reported confidence intervals: at n = 30 a non-significant test is
   not proof of no difference.
2. **Multiple linear regression** — price regressed on location, property,
   and capacity attributes (`id` and `host_id` are identifiers and are
   excluded). Held-out RMSE and MAE are reported from a seeded random
   70/30 train/test split, for both a price model and a log-price model.
   Regression and LASSO use the same predictor set (including `bedrooms`)
   and the shared cleaned modelling data produced by the same
   `rm.outlier` price rule as the t-tests; the headline held-out RMSE/MAE
   are dollar errors on that cleaned raw-price basis.
3. **LASSO regularisation** — seeded `glmnet` / `cv.glmnet` on a design matrix with
   proper dummy (one-hot) coding of the categorical columns (again
   excluding `id` and `host_id`), at `lambda.min` and `lambda.1se`.
4. **Classification** — LDA and k-nearest neighbours (k = 1 and k = 10,
   reported separately) predicting `room_type` from `price`, on seeded
   random 70/30 train/test splits, compared against a majority-class
   baseline.
5. **Clustering** — k-means (k = 2, 3, 4) on scaled `bedrooms` and `price`.

## Results

Re-run the script on the dataset described above to generate the current
numbers; this README does not restate results, so that the figures always
come from the code (held-out RMSE/MAE, classifier accuracies versus the
majority-class baselines, and the t-test confidence intervals are printed
by the script). The accompanying write-up is in
`Airbnb_project_paper.pdf`.

## How to run

Requirements: R, with the packages loaded at the top of the script
(`outliers`, `rcompanion`, `lars`, `glmnet`, `ggplot2`, `class`, `MASS`,
`factoextra`) — the script contains the corresponding `install.packages`
lines.

```sh
# from the repository root, with the data in place (see "Data" above)
Rscript Airbnb_project_code.r
```

Note: the script is interactive in places (regression diagnostic plots
prompt you to press Return in the console), so running it inside RStudio
or an interactive R session is more convenient than batch mode.

## Note

This is coursework. It is shared for educational / portfolio purposes and
has not been maintained as a production analysis; figures in the paper
were produced from an earlier version of the code.

## License

MIT — see [LICENSE](LICENSE).
