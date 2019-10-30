# Title: Improving Random Forest in Image Classification Using t-SNE and K-means Clustering
# Young-do Cho, 2019 

# 0. Configuration
## Computing Environment
# - CPU: Intel i5 4690
# - RAM: 24GB
# - OS: Ubuntu Desktop 18.04.3
# - R version: 3.6.1

## Install packages
install.packages("tidyverse") # Metapackage with lots of helpful functions
install.packages("ranger") # Fast Implementation of Random Forests
install.packages("Rtsne") # T-Distributed Stochastic Neighbor Embedding (t-SNE)

## Set packages
library(tidyverse) # ver. 1.2.1
library(ranger) # ver. 0.11.2
library(Rtsne) # ver. 0.15





# 1. Load data
## 1. MNIST
mnist_train <- read.csv("mnistTrainSet.csv")
mnist_test <- read.csv("mnistTestSet.csv")

mnist_train$label <- as.factor(mnist_train$label)
mnist_test$label <- as.factor(mnist_test$label)

str(mnist_train)
str(mnist_test)

## 2. fashion-MNIST
fashion_mnist_train <- read.csv("fashion-mnist_train.csv")
fashion_mnist_test <- read.csv("fashion-mnist_test.csv")

fashion_mnist_train$label <- as.factor(fashion_mnist_train$label)
fashion_mnist_test$label <- as.factor(fashion_mnist_test$label)

str(fashion_mnist_train)
str(fashion_mnist_test)

# 2. t-SNE
## 1. MNIST
mnist_all <- rbind(mnist_train, mnist_test)
set.seed(2019) # to always get the same result
mnist_tsne <- Rtsne(mnist_all[, -1], dims = 2, perplexity=30, check_duplicates = FALSE, verbose=TRUE, max_iter = 500)
tsne_2d_mnist_all <- as.data.frame(mnist_tsne$Y)
tsne_2d_mnist_train <- tsne_2d_mnist_all[1:60000, ]
tsne_2d_mnist_test <- tsne_2d_mnist_all[60001:70000, ]

### graph
set.seed(2019)
select <- sample(1:nrow(mnist_train), 6000)
selected_mnist_train <- mnist_train[select,]
selected_tsne_mnist_train <- tsne_2d_mnist_train[select, ]
colors <- rainbow(10)
names(colors) <- unique(selected_mnist_train$label)
par(mgp = c(2.5, 1,0))
plot(selected_tsne_mnist_train, t="n", main="tSNE (MNIST)", xlab="tSNE dimension 1", ylab="tSNE dimension 2")
text(selected_tsne_mnist_train, labels=selected_mnist_train$label, col=colors[selected_mnist_train$label])


## 2. fashion-MNIST
fashion_mnist_all <- rbind(fashion_mnist_train, fashion_mnist_test)
set.seed(2019) # to always get the same result
fashion_mnist_tsne <- Rtsne(fashion_mnist_all[, -1], dims = 2, perplexity=30, check_duplicates = FALSE, verbose=TRUE, max_iter = 500)
tsne_2d_fashion_mnist_all <- as.data.frame(fashion_mnist_tsne$Y)
tsne_2d_fashion_mnist_train <- tsne_2d_fashion_mnist_all[1:60000, ]
tsne_2d_fashion_mnist_test <- tsne_2d_fashion_mnist_all[60001:70000, ]

### graph
set.seed(2019)
select <- sample(1:nrow(fashion_mnist_train), 6000)
selected_fashion_mnist_train <- fashion_mnist_train[select,]
selected_tsne_fashion_mnist_train <- tsne_2d_fashion_mnist_train[select, ]
colors <- rainbow(10)
names(colors) <- unique(selected_fashion_mnist_train$label)
par(mgp = c(2.5, 1,0))
plot(selected_tsne_fashion_mnist_train, t="n", main="tSNE (MNIST)", xlab="tSNE dimension 1", ylab="tSNE dimension 2")
text(selected_tsne_fashion_mnist_train, labels=selected_fashion_mnist_train$label, col=colors[selected_fashion_mnist_train$label])





# 3. Experiment
## Expt.1) Random Forest 
### 1. MNIST
set.seed(2019)
model_m_expt1 <- ranger(label ~ ., data = mnist_train, importance = "impurity")
print(model_m_expt1)
print(model_m_expt1$prediction.error)

### 2. fashion-MNIST
set.seed(2019)
model_fm_expt1 <- ranger(label ~ ., data = fashion_mnist_train, importance = "impurity")
print(model_fm_expt1)
print(model_fm_expt1$prediction.error)



## Expt.2) K-means -> RF
### 1. MNIST
for(i in 2:50) {
    temp_m_train <- mnist_train
    set.seed(2019)
    kmeans_result <- kmeans(temp_m_train[, -1], centers = i, iter.max = 10000, nstart = 20)
    temp_m_train$cluster <- as.factor(kmeans_result$cluster)
    print(i)
    print("Expt 2. fit one model (just kmeans)")
    set.seed(2019)
    model <- ranger(label ~ ., data = temp_m_train, importance = "impurity")
    print(model)
    print(model$prediction.error)
}

### 2. fashion-MNIST
for(i in 2:50) {
    temp_fm_train <- fashion_mnist_train
    set.seed(2019)
    kmeans_result <- kmeans(temp_fm_train[, -1], centers = i, iter.max = 10000, nstart = 20)
    temp_fm_train$cluster <- as.factor(kmeans_result$cluster)
    print(i)
    print("Expt 2. fit one model (just kmeans)")
    set.seed(2019)
    model <- ranger(label ~ ., data = temp_fm_train, importance = "impurity")
    print(model)
    print(model$prediction.error)
}



## Expt.3) t-SNE + K-means -> RF
### 1. MNIST
for(i in 2:50) {
    temp_m_train <- mnist_train
    set.seed(2019)
    kmeans_result <- kmeans(tsne_2d_mnist_train[, -1], centers = i, iter.max = 10000, nstart = 20)
    temp_m_train$cluster <- as.factor(kmeans_result$cluster)
    print(i)
    print("Expt 2. fit one model (just kmeans)")
    set.seed(2019)
    model <- ranger(label ~ ., data = temp_m_train, importance = "impurity")
    print(model)
    print(model$prediction.error)
}

### 2. fashion-MNIST
for(i in 2:50) {
    temp_fm_train <- fashion_mnist_train
    set.seed(2019)
    kmeans_result <- kmeans(tsne_2d_fashion_mnist_train[, -1], centers = i, iter.max = 10000, nstart = 20)
    temp_fm_train$cluster <- as.factor(kmeans_result$cluster)
    print(i)
    print("Expt 2. fit one model (just kmeans)")
    set.seed(2019)
    model <- ranger(label ~ ., data = temp_fm_train, importance = "impurity")
    print(model)
    print(model$prediction.error)
}



## Expt.4) t-SNE + RF
### 1. MNIST
temp_m_train <- mnist_train
temp_m_train$tsne_X <- tsne_2d_mnist_train$V1
temp_m_train$tsne_Y <- tsne_2d_mnist_train$V2
set.seed(2019)
model_m_expt4 <- ranger(label ~ ., data = temp_m_train, importance = "impurity")
print(model_m_expt4)
print(model_m_expt4$prediction.error) # = 0.02301667

### 2. fashion-MNIST
temp_fm_train <- tsne_2d_fashion_mnist_train
temp_fm_train$tsne_X <- tsne_2d_fashion_mnist_train$V1
temp_fm_train$tsne_Y <- tsne_2d_fashion_mnist_train$V2
set.seed(2019)
model_fm_expt4 <- ranger(label ~ ., data = temp_fm_train, importance = "impurity")
print(model_fm_expt4)
print(model_fm_expt4$prediction.error) # = 0.1143333





# 4. Test
## Choose Expt.4 model for both data
### 1. MNIST
temp_m_test <- mnist_test
temp_m_test$tsne_X <- tsne_2d_mnist_test$V1
temp_m_test$tsne_Y <- tsne_2d_mnist_test$V2
pred <- predict(model_m_expt4, temp_m_test)$predictions
real <- temp_m_test$label
print(confusionMatrix(pred, real))

### 2. fashion-MNIST
temp_fm_test <- fashion_mnist_test
temp_fm_test$tsne_X <- tsne_2d_fashion_mnist_test$V1
temp_fm_test$tsne_Y <- tsne_2d_fashion_mnist_test$V2
pred <- predict(model_fm_expt4, temp_fm_test)$predictions
real <- temp_fm_test$label
print(confusionMatrix(pred, real))
