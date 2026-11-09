### 1 Hypothesis testing
# Data: place the listings CSV at data/airbnb_listings.csv (relative to this
# script / the repository root) before running. The data file is not included
# in this repository -- see README.md ("Data") for the source and download step.
nymalistings <- read.csv("data/airbnb_listings.csv", header = TRUE, na.strings = "") #Loading data

#commands to install the packages
install.packages("outliers") #package to detect the outliers
install.packages("rcompanion") #package to create normal distribution cure, histogram
install.packages("lars") #package to perform linear and lasso regression
install.packages("glmnet") #package to perform linear and lasso regression
install.packages("ggplot2") #package to plot graphs
install.packages("class") #package to perform classification
install.packages("MASS") #package to perform classification
install.packages("factoextra") #package to create clustering plots

library(outliers) #package to detect the outliers
library(rcompanion) #package to create normal distribution cure, histogram
library(lars) #package to perform linear and lasso regression
library(glmnet) #package to perform linear and lasso regression
library(ggplot2) #package to plot graphs
library(class) #package to perform classification
library(MASS) #package to perform classification
library(factoextra) #package to create clustering plots


nylistings <- subset(nymalistings, state=='NY'& bedrooms=='3') #Filtering data by state  and no. of bedrooms for NY
malistings <- subset(nymalistings, state=='MA' & bedrooms=='3') # Filtering data by state  and no. of bedrooms for MA

cleannylistings <- rm.outlier(nylistings$price, fill = FALSE, median = FALSE, opposite = FALSE) #Remove outliers NY
cleanmalistings <- rm.outlier(malistings$price, fill = FALSE, median = FALSE, opposite = FALSE) #Remove outliers MA

qqnorm(cleannylistings,main = "3 bedroom overnight Prices in NY", xlab = 'Price (in dollars)') # Check normality NY
qqline(cleannylistings,main = "3 bedroom overnight Prices in NY", xlab = 'Price (in dollars)') # Check normality NY
plotNormalHistogram(cleannylistings, main = "3 bedroom overnight Prices in NY", xlab = 'Price (in dollars)', xlim = c(10,900)) # Check normality NY


qqnorm(cleanmalistings, main = "3 bedroom overnight Prices in MA", xlab = 'Price (in dollars)') #Check normality MA
qqline(cleanmalistings, main = "3 bedroom overnight Prices in MA", xlab = 'Price (in dollars)') #Check normality MA
plotNormalHistogram(cleanmalistings, main = "3 bedroom overnight Prices in MA", xlab = 'Price (in dollars)', xlim = c(10,750)) #Check normality MA

set.seed(123) #Seed before sampling, so the t-test samples below are reproducible
nysample <- sample(cleannylistings,30) #Taking a random sample of 30 from NY, 3 bedroom listings

masample <- sample(cleanmalistings,30) #Taking a random sample of 30 from MA, 3 bedroom listings


# Interpretation note: with n = 30 per group these tests have low power.
# A non-significant result is NOT proof that there is no difference -- it only
# means the data do not rule one out. Report the confidence interval for the
# difference in means (printed below) and judge which differences remain
# plausible, rather than treating "p > 0.05" as "no difference".
tt_diff <- t.test(nysample, masample, alternative = 'two.sided') #Testing the difference between Mean prices of 3 bedrooms in NY and MA
tt_diff
tt_diff$conf.int #95% confidence interval for the NY - MA difference in mean price

var.test(nysample, masample, alternative = "two.sided") #Testing the difference in variances in 3 bedroom prices in NY and MA

tt_ny <- t.test(nysample, alternative = 'two.sided', mu=227) #Testing claim mean price in NY for 3 bedrooms = 227
tt_ny
tt_ny$conf.int #95% confidence interval for the NY mean price

tt_ma <- t.test(masample, alternative = 'two.sided', mu=338) #Testing claim mean price in MA for 3 bedrooms = 338
tt_ma
tt_ma$conf.int #95% confidence interval for the MA mean price

## Shared cleaned modelling data and held-out factor-level guard
# Sections 2 (regression) and 3 (LASSO) use the same price-outlier rule as
# the t-tests above: outliers::rm.outlier() on price with fill = FALSE,
# median = FALSE, opposite = FALSE. Both models therefore use raw prices on
# this one cleaned basis, rather than the t-tests cleaning price while the
# models silently used every raw row.
clean_model_prices <- rm.outlier(nymalistings$price, fill = FALSE, median = FALSE, opposite = FALSE)
modeling_data_clean <- nymalistings[nymalistings$price %in% clean_model_prices, ]
cat("Shared modelling data: kept", nrow(modeling_data_clean), "of", nrow(nymalistings), "rows after the shared price-outlier rule.\n")

# Held-out rows can contain categorical levels that never occur in training;
# predict.lm() and model.matrix() can then fail or produce incompatible
# columns. Align each categorical test column to the training levels:
# unseen levels become NA, and rows whose NA was introduced this way are
# dropped (with the count printed).
align_test_factor_levels <- function(train_data, test_data, categorical_columns) {
  unseen_rows <- rep(FALSE, nrow(test_data))
  for (column_name in categorical_columns) {
    if (!(column_name %in% names(train_data)) || !(column_name %in% names(test_data))) next
    train_values <- train_data[[column_name]]
    if (!is.factor(train_values) && !is.character(train_values)) next
    train_levels <- if (is.factor(train_values)) levels(droplevels(train_values)) else unique(as.character(train_values[!is.na(train_values)]))
    test_values <- as.character(test_data[[column_name]])
    unseen_rows <- unseen_rows | (!is.na(test_values) & !(test_values %in% train_levels))
    test_data[[column_name]] <- factor(test_values, levels = train_levels)
  }
  dropped_rows <- sum(unseen_rows)
  cat("Dropped", dropped_rows, "held-out rows with categorical levels not present in the training data.\n")
  test_data[!unseen_rows, , drop = FALSE]
}

###2 Multile Linear Regression

summary(nymalistings) #summary of the dataset represents the min, max, and quartile values

ggplot(nymalistings, aes(x=price)) + geom_histogram(binwidth=10) + labs(x="price") #plotting the minimum,max,1st quartile,3rd quartile,mean etc  for price variable

(l <- sapply(nymalistings, function(x) is.factor(x)))  #sapply function is used to determine which columns/variables are factors and return a vector

m <- data.frame(nymalistings[, l]) #function to create a data frame

ifelse(n <- sapply(m, function(x) length(levels(x))) == 1, "DROP", "NODROP") #This command is used to determine which of the variables in the dataset are factors or characters which take a single value for every input

which(sapply(m, function(x) length(unique(x))<2)) #command to calculate the number of level/length and return only unique value

# NOTE: id and host_id are identifiers, not price drivers, so they are NOT
# used as predictors in the regression (or in the LASSO matrix in section 3).
# The regression and LASSO now use the same predictor set, including bedrooms.
linearMod <- lm(price ~ street + neighbourhood_cleansed + city + state + zipcode + latitude + longitude + property_type + room_type + accommodates + bathrooms + bedrooms + beds + bed_type + review_scores_rating, data=modeling_data_clean) #function to implement linear regression for price(dependent variable) over other independent variables

summary(linearMod) #summary of the linear model

plot(linearMod) #function to plot the linear model ##continue pressing enter(return) button in the console window to view more plots. There are 4 plots related linear mod.

predict_price <- predict(linearMod) #predict() function to predict the values of the price variable and compare it with the original dataset
predict_price

residual_price <- data.frame(resid(linearMod)) #resid() function to calculate the residue, which is the difference between the original price data and the predicted value
format(residual_price, scientific=FALSE)

## Held-out evaluation of the regression (RMSE / MAE)
# The model above is fit on all cleaned rows, so its in-sample fit overstates
# how well it predicts new listings. Refit on a random training split and
# report RMSE and MAE on the held-out test rows. The headline RMSE/MAE below
# are dollar errors on the shared cleaned raw-price basis
# (modeling_data_clean); the log-price comparison uses the same cleaned split
# and is back-transformed to dollars. Price is right-skewed, so the log-price
# model is also fit and compared on the price scale (via exp(prediction);
# note this back-transformation is approximate and tends to under-predict).
set.seed(123) #Seed before the regression train/test split
reg_columns <- c("price", "street", "neighbourhood_cleansed", "city", "state", "zipcode", "latitude", "longitude", "property_type", "room_type", "accommodates", "bathrooms", "bedrooms", "beds", "bed_type", "review_scores_rating")
reg_data <- na.omit(modeling_data_clean[, reg_columns])
reg_train_idx <- sample(seq_len(nrow(reg_data)), size = floor(0.7 * nrow(reg_data)))
reg_train <- reg_data[reg_train_idx, ]
reg_test <- reg_data[-reg_train_idx, ]
reg_test <- align_test_factor_levels(reg_train, reg_test, setdiff(reg_columns, "price"))

reg_formula <- price ~ street + neighbourhood_cleansed + city + state + zipcode + latitude + longitude + property_type + room_type + accommodates + bathrooms + bedrooms + beds + bed_type + review_scores_rating
linearMod_train <- lm(reg_formula, data = reg_train)
pred_test <- predict(linearMod_train, newdata = reg_test)
rmse_test <- sqrt(mean((reg_test$price - pred_test)^2)) #held-out RMSE, in dollars
mae_test <- mean(abs(reg_test$price - pred_test)) #held-out MAE, in dollars
rmse_test
mae_test

# Log-price version (price > 0 only, as log(0) is undefined)
reg_train_pos <- reg_train[reg_train$price > 0, ]
reg_test_pos <- reg_test[reg_test$price > 0, ]
logMod_train <- lm(log(price) ~ street + neighbourhood_cleansed + city + state + zipcode + latitude + longitude + property_type + room_type + accommodates + bathrooms + bedrooms + beds + bed_type + review_scores_rating, data = reg_train_pos)
pred_log_test <- exp(predict(logMod_train, newdata = reg_test_pos)) #back-transformed to dollars
rmse_log_test <- sqrt(mean((reg_test_pos$price - pred_log_test)^2)) #held-out RMSE for the log-price model, in dollars
mae_log_test <- mean(abs(reg_test_pos$price - pred_log_test)) #held-out MAE for the log-price model, in dollars
rmse_log_test
mae_log_test

### 3. Lasso Regularization

# Predictor columns are selected by name. id and host_id are excluded (see
# the note in section 2), and price is the response. This is the same
# predictor set as the regression, including bedrooms, and DF starts from the
# same shared cleaned modelling data (modeling_data_clean).
lasso_predictors <- c("street", "neighbourhood_cleansed", "city", "state", "zipcode", "latitude", "longitude", "property_type", "room_type", "accommodates", "bathrooms", "bedrooms", "beds", "bed_type", "review_scores_rating")
DF <- modeling_data_clean[, c(lasso_predictors, "price")] #Created a dataframe

DF <- na.omit(DF) #omitted the null/missing values from the dataframe

y <- DF$price #price variable for lasso model

# Proper dummy (one-hot) coding for the categorical columns: model.matrix
# expands each factor/character column into 0/1 indicator columns and leaves
# numeric columns as they are. (The previous lapply(x, as.numeric) silently
# replaced categories with arbitrary integer level codes.) LASSO has no
# separate held-out design matrix in this script: this one shared Mx, built
# once from DF, is used for every cv.glmnet fold, so its columns cannot
# diverge between training and validation rows.
Mx <- model.matrix(price ~ ., data = DF)[, -1] #creating a matrix (drop the intercept column; glmnet adds its own)
set.seed(123) #Seed immediately before cv.glmnet, so its fold assignment is reproducible
lasso_model <- cv.glmnet(Mx , y, alpha = 1) #Setting up the model for lasso

lasso_model1 <- glmnet(Mx,y) #setting up the model

plot.cv.glmnet(lasso_model)  #Plot of cross valildation error according to log lambda values

fit <- glmnet(x=Mx, y=y, alpha = 1, lambda= lasso_model$lambda.min) #When lambda is minimum

fit$beta #diplay the important features

fit1 <- glmnet(x=Mx, y=y, alpha = 1, lambda= lasso_model$lambda.1se) #When lambda is maximum

fit1$beta #diplay the important features
plot.glmnet(lasso_model1) #Plot showing path of coefficient of variables against L1-norm as lambda varies


###4 Data Mining
###4.1 Classification
###4.1.1 LDA CLassification

##LDA Classification on MA dataset
data_ny <- subset(nymalistings, state == 'NY') #Data for New York
data_ma <- subset(nymalistings, state == 'MA') #Data for Massachusetts

# Random train/test splits (seeded, reproducible), replacing the old
# row-index splits (rows 1:2000 vs the rest), which depended on the row
# order of the file. Columns are selected by name, not by position.
# The same MA / NY splits are reused for KNN in section 4.1.2, so the
# classifiers are compared on identical held-out rows.
set.seed(123) #Seed before the train/test split
train_idx_ma <- sample(seq_len(nrow(data_ma)), size = floor(0.7 * nrow(data_ma)))
train_data <- data_ma[train_idx_ma, c("room_type", "price")]           #training data for MA
test_data <- data_ma[-train_idx_ma, c("room_type", "price")]       #testing data  for MA
test_data <- align_test_factor_levels(train_data, test_data, "room_type")

m1 <- lda(room_type ~ price, data = train_data) #lda model for MA dataset
plot(m1)  #lda plot for MA dataset
predict_lda <-predict(m1,test_data) #prediction based on test data for MA dataset
predict_lda$class #class variables for prediction
predict_lda$posterior #posterior probabilities based on predictions
table(test_data$room_type, predict_lda$class) #cross table to find accuracy

mean(test_data$room_type == predict_lda$class ) #find mean of matched data to find accuracy

# Majority-class baseline for MA: the accuracy of always predicting the
# most common room_type in the test set. A model is only useful if it
# beats this.
majority_baseline_ma <- max(table(test_data$room_type)) / length(test_data$room_type)
majority_baseline_ma

##LDA CLassification on NY dataset

set.seed(123) #Seed before the NY train/test split
train_idx_ny <- sample(seq_len(nrow(data_ny)), size = floor(0.7 * nrow(data_ny)))
train_data_ny <- data_ny[train_idx_ny, c("room_type", "price")]          #training data for NY
test_data_ny <- data_ny[-train_idx_ny, c("room_type", "price")]      #testing data for NY
test_data_ny <- align_test_factor_levels(train_data_ny, test_data_ny, "room_type")

m1_ny <- lda(room_type ~ price, data = train_data_ny) #fitting model creation
plot(m1_ny) #plot of fit model

predict_lda_ny <-predict(m1_ny,test_data_ny) #prediction on testing data -- NOTE: m1_ny (the NY model), not m1 (the MA model)
predict_lda_ny$class  #class variables i.e. predicted variables
predict_lda_ny$posterior #posterior probability of classes
table(test_data_ny$room_type, predict_lda_ny$class) #cross table for accuracy

mean( test_data_ny$room_type == predict_lda_ny$class ) #finding mean of matched values

# Majority-class baseline for NY (see the MA note above)
majority_baseline_ny <- max(table(test_data_ny$room_type)) / length(test_data_ny$room_type)
majority_baseline_ny

###4.1.2 K nearest Neighbor Classification
##KNN for MA dataset
# Reuses the seeded random split from section 4.1.1 (train_idx_ma), including
# the factor-aligned held-out rows in test_data, so KNN is evaluated on
# exactly the same test rows as LDA.

train_data_knn <- data_ma[train_idx_ma, "price", drop = FALSE] #training data for quantitative variables
test_data_knn <- test_data[, "price", drop = FALSE] #testing data for quantitative variables
train_data_ql <- data_ma[train_idx_ma, "room_type"] #training data for qualitative variable
test_data_ql <- test_data$room_type #testing data for qualitative variable

# k = 1 and k = 10 are kept in separate objects, so neither result
# overwrites the other (the previous code reused a single variable).
knn_model_k1 <- knn(data.frame(train_data_knn), data.frame(test_data_knn), train_data_ql , k = 1) #creating a knn model with k = 1
knn_model_k10 <- knn(data.frame(train_data_knn), data.frame(test_data_knn), train_data_ql , k = 10) #creating a knn model with k = 10

table(test_data_ql,knn_model_k1) #cross table to check accuracy, k = 1
mean(test_data_ql == knn_model_k1) #to find the matched values by the predicted system, k = 1

table(test_data_ql,knn_model_k10) #cross table to check accuracy, k = 10
mean(test_data_ql == knn_model_k10) #to find the matched values by the predicted system, k = 10

majority_baseline_ma #majority-class baseline for MA, from section 4.1.1 -- compare both k values against this

##KNN for NY dataset
# Reuses the seeded random split from section 4.1.1 (train_idx_ny), including
# the factor-aligned held-out rows in test_data_ny, so KNN is evaluated on
# exactly the same test rows as LDA.

train_data_knn_ny <- data_ny[train_idx_ny, "price", drop = FALSE] #training data for quantitative variables
test_data_knn_ny <- test_data_ny[, "price", drop = FALSE] #training data for quantitative variables
train_data_ql_ny <- data_ny[train_idx_ny, "room_type"]  #training data for qualitative variables
test_data_ql_ny <- test_data_ny$room_type #training data for qualitative variables

knn_model_ny_k1 <- knn(data.frame(train_data_knn_ny), data.frame(test_data_knn_ny), train_data_ql_ny , k = 1) #creating a knn model with k = 1
knn_model_ny_k10 <- knn(data.frame(train_data_knn_ny), data.frame(test_data_knn_ny), train_data_ql_ny , k = 10) #creating a knn model with k = 10

table(test_data_ql_ny,knn_model_ny_k1) #cross table to check accuracy, k = 1
mean(test_data_ql_ny == knn_model_ny_k1) #to find the matched values by the predicted system, k = 1

table(test_data_ql_ny,knn_model_ny_k10) #cross table to check accuracy, k = 10
mean(test_data_ql_ny == knn_model_ny_k10) #to find the matched values by the predicted system, k = 10

majority_baseline_ny #majority-class baseline for NY, from section 4.1.1 -- compare both k values against this

###4.2 Clustering
##Kmeans clustering on MA

data_kmeans_ma <- data_ma[,c("bedrooms", "price")] #data for kmeans(bedrooms and price variables)
data_kmeans_ma <- na.omit(data_kmeans_ma) #removing NA values from dataset
data_kmeans_ma_scaled <- scale(data_kmeans_ma)  #normalizing data using scale function
set.seed(123)  #setting seed for kmeans
### for k =2
final <- kmeans(data_kmeans_ma_scaled, 2, nstart = 25) #Kmeans clustering to make 3 clusters
attributes(final) #to get the attributes of final variable
fviz_cluster(final, data = data_kmeans_ma_scaled) #plot for clustering

###for k =3
final <- kmeans(data_kmeans_ma_scaled, 3, nstart = 25)  #Kmeans clustering to make 3 clusters
attributes(final) #to get the attributes of final variable
fviz_cluster(final, data = data_kmeans_ma_scaled)  #plot for clustering
### for k = 4
final <- kmeans(data_kmeans_ma_scaled, 4, nstart = 25)  #Kmeans clustering to make 3 clusters
attributes(final) #to get the attributes of final variable
fviz_cluster(final, data = data_kmeans_ma_scaled)   #plot for clustering


##Kmeans clustering on NY

data_kmeans_ny <- data_ny[,c("bedrooms", "price")] #data for kmeans(bedrooms and price variables)
data_kmeans_ny <- na.omit(data_kmeans_ny) #removing NA values from dataset
data_kmeans_ny_scaled <- scale(data_kmeans_ny)  #normalizing data using scale function
set.seed(123) #setting seeds for k means
## for k =2
final_ny <- kmeans(data_kmeans_ny_scaled, 2, nstart = 25)  #Kmeans clustering to make 2 clusters
attributes(final_ny) #to get the attributes of final variable
fviz_cluster(final_ny, data = data_kmeans_ny_scaled) #plot for clustering

## for k =3
final_ny <- kmeans(data_kmeans_ny_scaled, 3, nstart = 25)  #Kmeans clustering to make 3 clusters
attributes(final_ny) #to get the attributes of final variable
fviz_cluster(final_ny, data = data_kmeans_ny_scaled) #plot for clustering

## for k =4
final_ny <- kmeans(data_kmeans_ny_scaled, 4, nstart = 25)  #Kmeans clustering to make 4 clusters
attributes(final_ny) #to get the attributes of final variable
fviz_cluster(final_ny, data = data_kmeans_ny_scaled)   #plot for clustering

###END###
