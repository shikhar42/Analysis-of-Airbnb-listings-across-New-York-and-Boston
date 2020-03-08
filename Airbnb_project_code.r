### 1 Hypothesis testing
nymalistings <- read.csv(file.choose(), header = TRUE, na.string = "") #Loading data

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

nysample <- sample(cleannylistings,30) #Taking a random sample of 30 from NY, 3 bedroom listings 

masample <- sample(cleanmalistings,30) #Taking a random sample of 30 from MA, 3 bedroom listings 


t.test(nysample, masample, alternative = 'two.sided') #Testing the difference between Mean prices of 3 bedrooms in NY and MA

var.test(nysample, masample, alternative = "two.sided") #Testing the difference in variances in 3 bedroom prices in NY and MA

t.test(nysample, alternative = 'two.sided', mu=227) #Testing claim mean price in NY for 3 bedrooms = 227

t.test(masample, alternative = 'two.sided', mu=338) #Testing claim mean price in MA for 3 bedrooms = 338 

###2 Multile Linear Regression

summary(nymalistings) #summary of the dataset represents the min, max, and quartile values

ggplot(nymalistings, aes(x=price)) + geom_histogram(binwidth=10) + labs(x="price") #plotting the minimum,max,1st quartile,3rd quartile,mean etc  for price variable 

(l <- sapply(nymalistings, function(x) is.factor(x)))  #sapply function is used to determine which columns/variables are factors and return a vector

m <- data.frame(nymalistings[, l]) #function to create a data frame

ifelse(n <- sapply(m, function(x) length(levels(x))) == 1, "DROP", "NODROP") #This command is used to determine which of the variables in the dataset are factors or characters which take a single value for every input

which(sapply(m, function(x) length(unique(x))<2)) #command to calculate the number of level/length and return only unique value

linearMod <- lm(price ~ id + host_id + street + neighbourhood_cleansed + city + state + zipcode + latitude + longitude + property_type + room_type + accommodates + bathrooms + beds +bed_type + review_scores_rating, data=nymalistings) #function to implement linear regression for price(dependent variable) over other independent variables 

summary(linearMod) #summary of the linear model

plot(linearMod) #function to plot the linear model ##continue pressing enter(return) button in the console window to view more plots. There are 4 plots related linear mod.

predict_price <- predict(linearMod) #predict() function to predict the values of the price variable and compare it with the original dataset
predict_price

residual_price <- data.frame(resid(linearMod)) #resid() function to calculate the residue, which is the difference between the original price data and the predicted value
format(residual_price, scientific=FALSE)

### 3. Lasso Regularization

DF <-data.frame(nymalistings$id, nymalistings$host_id, nymalistings$street, nymalistings$neighbourhood_cleansed, nymalistings$city, nymalistings$state, nymalistings$zipcode, nymalistings$latitude, nymalistings$longitude, nymalistings$property_type, nymalistings$room_type, nymalistings$accommodates,nymalistings$bathrooms, nymalistings$bedrooms,nymalistings$beds, nymalistings$bed_type, nymalistings$review_scores_rating, nymalistings$price) #Created a dataframe 

DF <- na.omit(DF) #omitted the null/missing values from the dataframe 

x<- data.frame(DF$nymalistings.id, DF$nymalistings.host_id, DF$nymalistings.street, DF$nymalistings.neighbourhood_cleansed, DF$nymalistings.city, DF$nymalistings.state, DF$nymalistings.zipcode, DF$nymalistings.latitude, DF$nymalistings.longitude, DF$nymalistings.property_type, DF$nymalistings.room_type, DF$nymalistings.accommodates, DF$nymalistings.bathrooms, DF$nymalistings.bedrooms, DF$nymalistings.beds, DF$nymalistings.bed_type, DF$nymalistings.review_scores_rating) #Combining data together of all variables
y <- DF$nymalistings.price #price variable for lasso model

Mx <- as.matrix(as.data.frame(lapply(x, as.numeric))) #creating a matrix
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

train_data<- data_ma[1:2000,c(12,18)]           #training data for MA
test_data <- data_ma[2001:3583, c(12,18)]       #testing data  for MA

m1 <- lda(room_type ~ price, data = train_data) #lda model for MA dataset
plot(m1)  #lda plot for MA dataset
predict_lda <-predict(m1,test_data) #prediction based on test data for MA dataset
predict_lda$class #class variables for prediction
predict_lda$posterior #posterior probabilities based on predictions 
table(test_data$room_type, predict_lda$class) #cross table to find accuracy

mean(test_data$room_type == predict_lda$class ) #find mean of matched data to find accuracy

##LDA CLassification on NY dataset

train_data_ny<- data_ny[1:2000,c(12,18)]          #training data for NY
test_data_ny <- data_ny[2001:3999, c(12,18)]      #testing data for NY
 
m1_ny <- lda(room_type ~ price, data = train_data_ny) #fitting model creation 
plot(m1_ny) #plot of fit model

predict_lda_ny <-predict(m1,test_data_ny) #prediction on testing data
predict_lda_ny$class  #class variables i.e. predicted variables
predict_lda_ny$posterior #posterior probability of classes
table(test_data_ny$room_type, predict_lda_ny$class) #cross table for accuracy
 
mean( test_data_ny$room_type == predict_lda_ny$class ) #finding mean of matched values

###4.1.2 K nearest Neighbor Classification
##KNN for MA dataset

train_data_knn <- data_ma[1:2000,c(18)] #training data for quantitative variables
test_data_knn <- data_ma[2001:3583,c(18)]# testing data for quantitative variables
train_data_ql <-data_ma[1:2000,c(12)] #training data for qualitative variable
test_data_ql <-data_ma[2001:3583,c(12)]#testing data for qualitative variable
knn_model <- knn(data.frame(train_data_knn), data.frame(test_data_knn), train_data_ql , k = 1) #creating a knn model with k = 1

knn_model <- knn(data.frame(train_data_knn), data.frame(test_data_knn), train_data_ql , k = 10) #creating a knn model with k = 10

table(test_data_ql,knn_model) #cross table to check accuracy

mean(test_data_ql == knn_model) #to find the matched values by the predicted system
##KNN for NY dataset

train_data_knn_ny <- data_ny[1:2000,c(18)] #training data for quantitative variables
test_data_knn_ny <- data_ny[2001:3999,c(18)] #training data for quatitative variables
train_data_ql_ny <-data_ny[1:2000,c(12)]  #training data for qualitative variables
test_data_ql_ny <-data_ny[2001:3999,c(12)]#training data for qualitative variables
knn_model_ny <- knn(data.frame(train_data_knn_ny), data.frame(test_data_knn_ny), train_data_ql_ny , k = 1) #creating a knn model with k = 1

knn_model_ny <- knn(data.frame(train_data_knn_ny), data.frame(test_data_knn_ny), train_data_ql_ny , k = 10) #creating a knn model with k = 10
table(test_data_ql_ny,knn_model_ny) #cross table to check accuracy

mean(test_data_ql_ny == knn_model_ny) #to find the matched values by the predicted system

###4.2 Clustering
##Kmeans clustering on MA

data_kmeans_ma <- data_ma[,c(15,18)] #data for kmeans(bedrooms and price variables)
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

data_kmeans_ny <- data_ny[,c(15,18)] #data for kmeans(bedrooms and price variables)
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

