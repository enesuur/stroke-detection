# Importing libraries and installing necessary packages.
install.packages(c("caret", "skimr", "RANN","randomForest","naivebayes","ROSE","mice","pROC","reshape"))
library(caret)
library(skimr)
library(RANN)
library(randomForest)
library(naivebayes)
library(ggplot2)
library(dplyr)
library(tidyr)
library(corrplot)
library(car)
library(pROC)
library(ROSE)
library(mice)
library(stats)
library(cluster)
library(reshape2)

# Step 1
# https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
# This dataset is utilized to predict stroke likelihood based on input parameters such as gender, age, 
# various diseases, and smoking status.
# Each row in the data provides pertinent information about the patient.
# Reading the CSV file and determining the response variable.
data <- read.csv("healthcare-dataset-stroke-data.csv")
response_variable <- "stroke"

# Step 2
# Displaying basic summary statistics for our dataset.
summary(data)

# Descriptive stats for each column.
descriptive_statistics <- skim_to_wide(data)
head(data,n=10)
str(data)
print(descriptive_statistics)

# Step 3.
# Data Distribution.
# Histogram for age.
ggplot(data, aes(x = age)) +
  geom_histogram(binwidth = 5, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Age", x = "Age", y = "Frequency")

# Boxplot for glucose level.
ggplot(data, aes(x = stroke, y = avg_glucose_level, fill = factor(stroke))) +
  geom_boxplot() +
  labs(title = "Boxplot of Glucose Level by Stroke", x = "Stroke", y = "Avg Glucose Level")

# Boxplot for BMI.
ggplot(data, aes(x = stroke, y = bmi, fill = factor(stroke))) +
  geom_boxplot() +
  labs(title = "Boxplot of BMI by Stroke", x = "Stroke", y = "Bmi")

# Categorical Features.
# Bar plot for gender distribution.
ggplot(data, aes(x = gender, fill = gender)) +
  geom_bar() +
  labs(title = "Gender Distribution", x = "Gender", y = "Count")

# Bar plot for work type distribution
ggplot(data, aes(x = work_type, fill = work_type)) +
  geom_bar() +
  labs(title = "Work Type Distribution", x = "Work Type", y = "Count")

# Geographical Analysis.
ggplot(data, aes(x = Residence_type, fill = factor(stroke))) +
  geom_bar(position = "dodge") +
  labs(title = "Stroke Distribution by Residence Type", x = "Residence Type", y = "Count")

# Bar plot for work type distribution
ggplot(data, aes(x = work_type, fill = work_type)) +
  geom_bar() +
  labs(title = "Work Type Distribution", x = "Work Type", y = "Count")

# Work Type and Stroke.
ggplot(data, aes(x = work_type, fill = factor(stroke))) +
  geom_bar(position = "dodge") +
  labs(title = "Stroke Distribution by Work Type", x = "Work Type", y = "Count")

# Hypertension and Heart Disease.
ggplot(data, aes(x = hypertension, fill = factor(stroke))) +
  geom_bar(position = "dodge") +
  labs(title = "Stroke Distribution by Hypertension", x = "Hypertension", y = "Count")

ggplot(data, aes(x = heart_disease, fill = factor(stroke))) +
  geom_bar(position = "dodge") +
  labs(title = "Stroke Distribution by Heart Disease", x = "Heart Disease", y = "Count")

# Geographical Analysis.
ggplot(data, aes(x = Residence_type, fill = factor(stroke))) +
  geom_bar(position = "dodge") +
  labs(title = "Stroke Distribution by Residence Type", x = "Residence Type", y = "Count")


# It prints that there is not a missing value in my data set but some columns has "N/A", we need to fix that.
data[data == "N/A"] <- NA

# Determining a seed value for controlling randomness.
set.seed(123)

# Using createDataPartition function for splitting data as training_data and test_data
# %80 -> train_data
# %20 -> test_data 
split_indices <- createDataPartition(data$stroke, p = 0.8, list = FALSE)

# Train Data
train_data <- data[split_indices, ]

# Train data with missing values
train_missing_values <- data[split_indices, ]

# Test Data
test_data <- data[-split_indices, ]

# Test data with missing values
test_missing_values <- data[-split_indices, ]

# Making stroke variable as factor in both data.
train_data$stroke <- factor(train_data$stroke)
test_data$stroke <- factor(test_data$stroke, levels = levels(train_data$stroke))
train_data$bmi <- as.numeric(train_data$bmi)
test_data$bmi <- as.numeric(test_data$bmi)


# Step 8
# X

# Y.a
# Checking for missing values.
missing_values <- colSums(is.na(data))
print("Missing Values:")
print(missing_values)
# There are some missing values in BMI.
# There are 42 missing values in BMI column.

# Being sure there is nothing left from "N/A" problem.
anyNA(data)
# It confirms we have missing values in our dataset.


# Creating model with K-NN for filling up missing values. 
preProcess_missingdata_model <- preProcess( train_data, method = "knnImpute")
train_data <- predict(preProcess_missingdata_model, newdata = train_data)
preProcess_missingdata_model <- preProcess( test_data, method = "knnImpute")
test_data <- predict(preProcess_missingdata_model, newdata = test_data)
anyNA(train_data)
# FALSE
anyNA(test_data)
# FALSE
anyNA(train_missing_values)
# TRUE
anyNA(test_missing_values)
# TRUE

# Y.b
# Applying random forest model for imputed dataset.
rf_model_imputed <- train(stroke ~ ., data = train_data, method = "rf")

# Making predictions for our model.
rf_predicted_imputed <- predict(rf_model_imputed, test_data)


# Creating confusing matrix.
rf_conf_matrix_imputed <- confusionMatrix(reference = test_data$stroke, data = rf_predicted_imputed, mode = 'everything', positive = '1')

print(rf_conf_matrix_imputed)

# There is 0.7722  accuracy in our model. For better results and training model missing values needs to be
# imputed.

print("----- Original data has missing values -----")
head(train_missing_values)

print("----- Original data is without missing values -----")
head(train_data)

# Checking our data wheter is imbalanced dataset.
# Checking class imbalance in stroke.
table(data$stroke)

# Z.A
# Checking wheter my dataset is balanced.
# First,let's observe the data by seeing first 6 rows.
head(data)
table(data$stroke)
prop.table(table(data$stroke))
# According to the results we can clearly see our dataset is imbalanced.
# 0.3759398 No stroke proportion.
# 0.6240602 Stroke proportion.

# Applying oversampling and undersampling to our data for making it balanced.
n_new <- nrow(data)
fraction_positive_new <- 0.50
sampling_result <- ovun.sample(as.formula(paste(response_variable, "~ .")),
                               data = train_data,
                               method = "both",
                               N = n_new,
                               p = fraction_positive_new,
                               seed = 2018)
imbalanced_data <- train_data
balanced_data <- sampling_result$data
balanced_train_data <- balanced_data[split_indices, ]
# Checking the results of oversampling and undersampling.
# Data has balanced.
table(balanced_data$stroke)
prop.table(table(balanced_data$stroke))

# Data has no balance.
table(imbalanced_data$stroke)
prop.table(table(imbalanced_data$stroke))


# Z.b
# Creating a naivebayes model for imbalanced data.
naivebayes_model_imbalanced <- train(stroke ~ ., data = imbalanced_data, method = "naive_bayes")

# Making prediction on test data.
naivebayes_predicted_imbalanced <- predict(naivebayes_model_imbalanced, newdata = test_data)

# Creating a confusion matrix for results.
conf_matrix_naivebayes_imbalanced <- confusionMatrix(reference = test_data$stroke, data = naivebayes_predicted_imbalanced, mode = 'everything')

# Printing confusing matrix.
print(conf_matrix_naivebayes_imbalanced)

# Creating a naivebayes model for balanced data.
naivebayes_model_balanced <- train(stroke ~ ., data = balanced_data, method = "naive_bayes")

# Making prediction on test data.
naivebayes_predicted_balanced <- predict(naivebayes_model_balanced, newdata = test_data)

# Creating a confusion matrix for results
conf_matrix_naivebayes_balanced <- confusionMatrix(reference = test_data$stroke, data = naivebayes_predicted_balanced)

# Printing confusing matrix.
print(conf_matrix_naivebayes_balanced)

# For imbalanced data.
accuracy_imbalanced <- conf_matrix_naivebayes_imbalanced$overall["Accuracy"]
precision_imbalanced <- conf_matrix_naivebayes_imbalanced$byClass["Pos Pred Value"]
recall_imbalanced <- conf_matrix_naivebayes_imbalanced$byClass["Sensitivity"]
f1_score_imbalanced <- conf_matrix_naivebayes_imbalanced$byClass["F1"]

# For balanced data.
accuracy_balanced <- conf_matrix_naivebayes_balanced$overall["Accuracy"]
precision_balanced <- conf_matrix_naivebayes_balanced$byClass["Pos Pred Value"]
recall_balanced <- conf_matrix_naivebayes_balanced$byClass["Sensitivity"]
f1_score_balanced <- conf_matrix_naivebayes_balanced$byClass["F1"]

# Printing the performance metrics.
# Performance Metrics for Naive Bayes Model on Imbalanced Data
cat("\nPerformance Metrics for Naive Bayes Model on Imbalanced Data:\n")
print(paste("Accuracy:", accuracy_imbalanced))
print(paste("Precision:", precision_imbalanced))
print(paste("Recall:", recall_imbalanced))
print(paste("F1 Score:", f1_score_imbalanced))

# Performance Metrics for Naive Bayes Model on Balanced Data
cat("\nPerformance Metrics for Naive Bayes Model on Balanced Data:\n")
print(paste("Accuracy:", accuracy_balanced))
print(paste("Precision:", precision_balanced))
print(paste("Recall:", recall_balanced))
print(paste("F1 Score:", f1_score_balanced))
# The Naive Bayes model trained on the imbalanced dataset achieved a higher accuracy of 79.75%,
# with better precision at 72.73%, indicating strong overall performance in predicting outcomes. 
# However, its recall rate was lower at 38.10%, indicating a limitation in identifying all actual positive cases.
# In contrast, the model trained on the balanced dataset had a slightly lower accuracy of 78.48% and reduced precision
# at 66.67%, while maintaining the same recall rate of 38.10%. The choice between these models depends on the 
# specific task requirements, weighing the importance of precision versus recall in the given context.
# In that case i would choose imbalanced.*


# Correlation Analysis.
# Checking data types of selected columns because some of the columns might have missing values or non-numeric values.
str(train_data[, c("age", "hypertension", "heart_disease", "avg_glucose_level","bmi")])

# Converting selected columns to numeric.
train_data[, c("age", "hypertension", "heart_disease", "avg_glucose_level","bmi")] <-
  lapply(train_data[, c("age", "hypertension", "heart_disease", "avg_glucose_level","bmi")], as.numeric)

# Checking data types again for validation our process.
str(train_data[, c("age", "hypertension", "heart_disease", "avg_glucose_level","bmi")])

# Step 4
# Correlation calculation.
correlation_matrix <- cor(train_data[, c("age", "hypertension", "heart_disease", "avg_glucose_level","bmi")], use = "complete.obs")
print(correlation_matrix)
# Age and Hypertension (0.23836611): There is a moderate positive correlation. This suggests that as age increases, 
# the risk of hypertension also tends to increase.
# Age and Heart Disease (0.26125534): Again, there is a moderate positive correlation. Increasing age may also 
# increase the risk of heart disease.
# Average Glucose Level and BMI (0.28461083): This indicates a moderate positive correlation. 
# As BMI (Body Mass Index) increases, the average glucose level might also tend to increase.
# There is no significant correlation between heart disease and BMI.
# Heart Disease and BMI (-0.02866801): This value is almost zero and negative, indicating that there
# indicators in the dataset.

# Visualization of Correlation Matrix.
corrplot(correlation_matrix, method = "color")


# Step 5
# Features for PCA.
features_for_pca <- c("age", "hypertension", "heart_disease", "avg_glucose_level","bmi")
# Selecting columns for PCA.
data_for_pca <- train_data[, features_for_pca]
# Normalizing the data.
scaled_data <- scale(data_for_pca)
# Applying PCA.
pca_result <- prcomp(scaled_data, scale. = TRUE)
# Printing result of PCA.
print(pca_result)

#  PC1 is primarily influenced by age, hypertension, heart disease, and average glucose level, 
# with higher values in these variables contributing more to its variance. PC2 captures a pattern where age and heart
# is ease are positively associated, while hypertension and BMI exhibit a negative association. Overall, PC1 
# emphasizes age and health conditions, while PC2 highlights specific interplays between health indicators.

# Scree Plot for visualising the PCA.
scree_plot <- ggplot() +
  geom_line(aes(x = 1:length(pca_result$sdev), y = pca_result$sdev^2 / sum(pca_result$sdev^2)), color = "blue") +
  geom_point(aes(x = 1:length(pca_result$sdev), y = pca_result$sdev^2 / sum(pca_result$sdev^2)), color = "red") +
  labs(title = "Scree Plot for PCA",
       x = "Principal Component",
       y = "Proportion of Variance Explained") +
  theme_minimal()

# Biplot.
biplot_data <- as.data.frame(pca_result$x[, 1:2])
biplot_data$Stroke <- train_data$stroke

biplot <- ggplot(data = biplot_data, aes(x = PC1, y = PC2, color = factor(Stroke))) +
  geom_point(alpha = 0.7) +
  labs(title = "Biplot for the First Two Principal Components",
       x = "Principal Component 1",
       y = "Principal Component 2",
       color = "Stroke") +
  theme_minimal()

# Displaying the plots for our PCA.
print(scree_plot)
print(biplot)

# Step 6
# Fitting Logistic Regression model.
logistic_model <- glm(stroke ~ age + hypertension + heart_disease + avg_glucose_level + bmi,
                      data = train_data, 
                      family = "binomial")

# Summary of the Logistic Regression model
summary(logistic_model)

# Predictions on the test set
predictions <- predict(logistic_model, newdata = test_data, type = "response")
# Calculating accuracy for our logistic model.
binary_predictions <- ifelse(predictions > 0.5, 1, 0)
accuracy <- sum(binary_predictions == test_data$stroke) / length(test_data$stroke)
print(paste("Accuracy:", accuracy))

# Visualising of the logistic regression model and plotting the ROC curve. 
roc_curve <- roc(test_data$stroke, predictions)
plot(roc_curve, main = "ROC Curve", col = "blue", lwd = 2)
# AUC(Area under the curve) score.
auc_score <- auc(roc_curve)
print(paste("AUC Score:", auc_score))

# Calculating performance scores by using our logistic regression model.
predicted_values <- predict(logistic_model, newdata = test_data, type = "response")
predicted_labels <- ifelse(predicted_values > 0.5, 1, 0)

# Confusion Matrix.
conf_matrix <- confusionMatrix(as.factor(predicted_labels), as.factor(test_data$stroke))

# Extracting performance metrics.
accuracy <- conf_matrix$overall["Accuracy"]
precision <- conf_matrix$byClass["Sensitivity"]
recall <- conf_matrix$byClass["Sensitivity"]
specificity <- conf_matrix$byClass["Specificity"]
f1_score <- conf_matrix$byClass["F1"]

# Printing performance metrics that we have calculated above.
print(paste("Accuracy:", accuracy))
print(paste("Precision:", precision))
print(paste("Recall:", recall))
print(paste("Specificity:", specificity))
print(paste("F1 Score:", f1_score))

# The logistic regression model achieved an accuracy of 74.68%, with precision and recall both at 66.67%. 
# Specificity is 77.59%, indicating a better ability to identify negative instances. The F1 Score, balancing 
# precision and recall, is 58.33%, suggesting room for improvement in handling false positives and negatives.

# Step 7
# Selecting relevant features for clustering
features_for_kmeans <- c("age", "hypertension", "heart_disease", "avg_glucose_level","bmi")

# Creating a new data frame with selected features.
data_for_kmeans <- train_data[, features_for_kmeans]

# Normalizing the data.
scaled_data_kmeans <- scale(data_for_kmeans)

# Applying K-Means clustering
kmeans_result <- kmeans(scaled_data_kmeans, centers = 5)
pca_result_kmeans <- prcomp(scaled_data_kmeans)
pca_data_kmeans <- data.frame(pca_result_kmeans$x[,1:2])
pca_data_kmeans$cluster <- kmeans_result$cluster

ggplot(pca_data_kmeans, aes(x = PC1, y = PC2, color = as.factor(cluster))) +
  geom_point(alpha = 0.5) +
  theme_minimal() +
  labs(title = "K-Means Clustering with PCA",
       x = "Principal Component 1",
       y = "Principal Component 2",
       color = "Cluster")

# Select relevant features for hierarchical clustering
features_for_hierarchical <- c("age", "hypertension", "heart_disease", "avg_glucose_level","bmi")
data_for_hierarchical <- train_data[, features_for_hierarchical]

# Normalizing the data.
scaled_data_hierarchical <- scale(data_for_hierarchical)

# Apply hierarchical clustering
hierarchical_result <- hclust(dist(scaled_data_hierarchical))

# Choosing a specific number of clusters and get clustering results.
cutree_result <- cutree(hierarchical_result, 5)

# Visualizing clustering results.
ggplot(train_data, aes(x = age, y = avg_glucose_level, color = factor(cutree_result))) +
  geom_point() +
  labs(title = "Agglomerative Clustering Results", x = "Age", y = "Avg Glucose Level") +
  theme_minimal()

# For K-Means clustering
silhouette_kmeans <- silhouette(kmeans_result$cluster, dist(scaled_data_kmeans))

# For Agglomerative Hierarchical Clustering
silhouette_hierarchical <- silhouette(cutree_result, dist(scaled_data_hierarchical))


# Average Silhouette Score for K-Means
avg_silhouette_kmeans <- mean(silhouette_kmeans[, "sil_width"])

# Average Silhouette Score for Agglomerative Hierarchical Clustering
avg_silhouette_hierarchical <- mean(silhouette_hierarchical[, "sil_width"])


# Compare
print("Average Silhouette Score for K-Means:")
print(avg_silhouette_kmeans)
print("Average Silhouette Score for Hierarchical Clustering:")

# K-Means has 0.3166701 score Hiearchical Clustering has 0.2547413 score.
# K-means is more suitable for the data that we are working on.

# Step 8
# a
# Training Random Forest model
rf_model <- train(stroke ~ ., data = train_data, method = "rf")

# Making predictions on the test set.
rf_predicted <- predict(rf_model, newdata = test_data)

# Calculating confusion matrix.
rf_conf_matrix <- confusionMatrix(reference = test_data$stroke, data = rf_predicted, mode = 'everything')

# Printing the results.
print(rf_conf_matrix)

# Converting confusion matrix to a table for ggplot
conf_matrix_table <- as.table(rf_conf_matrix$table)

# Melting the table for ggplot
melted_conf_matrix <- melt(conf_matrix_table)


# We are using ggplot for drawing heatmap from confusion matrix.
ggplot(data = melted_conf_matrix, aes(x = Prediction , y = Reference, fill = value)) +
  geom_tile() + # ısı haritası katmanı
  scale_fill_gradient(low = "white", high = "red") + # renk skalası
  geom_text(aes(label = value), color = "black") + # değerlerin üzerine yazdırılması
  labs(x = "Predicted", y = "Actual", fill = "Count") + # etiketler
  theme_minimal() + # tema
  ggtitle("Confusion Matrix Heatmap for Random Forest Model") # başlık


# b
# Creating a naivebayes model for 2nd classification algorithm.
naivebayes_model <- train(stroke ~ ., data = train_data, method = "naive_bayes")

# Making prediction on test data.
naivebayes_predicted <- predict(naivebayes_model, newdata = test_data)

# Creating a confusion matrix for results.
conf_matrix_naivebayes <- confusionMatrix(reference = test_data$stroke, data = naivebayes_predicted, mode = 'everything')

# Printing confusing matrix.
print(conf_matrix_naivebayes)

# Between Random Forest and Naive Bayes models, both have an accuracy of 0.7975. 
# Random Forest shows better balance in sensitivity and specificity, with a higher F1 score, 
# indicating a more balanced performance. Naive Bayes, while having higher specificity, falls short in sensitivity,
# leading to a lower F1 score. Random Forest is generally preferable for a more balanced overall performance..

ggplot(train_data, aes(x = bmi, fill = stroke)) +
  geom_density(alpha = 0.5) +
  labs(title = "Variable Distribution by Class", x = "BMI", y = "Density")

print(pca_result)

# X
# Selecting the first two principal components.
pca_features <- pca_result$x[, 1:2]

# Adding the first two principal components to the original data.
train_data_with_pca <- cbind(train_data, pca_features)

# Fitting Logistic Regression model on original data.
logistic_model_original <- glm(stroke ~ ., data = train_data, family = "binomial")

# Fitting Logistic Regression model on data with principal components.
logistic_model_pca <- glm(stroke ~ ., data = train_data_with_pca, family = "binomial")


# Summary of the Logistic Regression models.
print("--- Results with original data. ---")
summary(logistic_model_original)

print("--- Results with components. ---")
summary(logistic_model_pca)



