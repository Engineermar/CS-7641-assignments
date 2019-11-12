setwd("~/Documents/Projects/R/ML/CompareModels")

for (package in c('caret', 'rpart', 'rpart.plot', 'gbm')) {
  if (!require(package, character.only=T, quietly=T)) {
    install.packages(package)
    library(package, character.only=T)
  }
}


#Download files
url="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(url, "pml-training.csv",method='curl')

url="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url, "pml-testing.csv",method='curl')

#Read,view and clean datasets
# Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate. The exercises were performed by six male participants aged between 20-28 years, with little weight lifting experience. We made sure that all participants could easily simulate the mistakes in a safe and controlled manner by using a relatively light dumbbell (1.25kg).
# 
# Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3isa6f0Ds

training<-read.csv("pml-training.csv",head=T, na.string=c("NA","#DIV/0!", ""))
testing<-read.csv("pml-testing.csv",head=T,na.string=c("NA","#DIV/0!", ""))

# Delete columns with more than 20% missing values
training<-training[ , apply(training , 2 , function (x)  sum(is.na(x)) < 0.2 *nrow(training)) ]
dim(training)

testing<-testing[ , apply(testing , 2 , function (x)  sum(is.na(x)) < 0.2 *nrow(testing)) ]
dim(testing)

head(training)[1:10]

head(testing)[1:10]

# Columns 1-7 we can delete too
training <-training[,-c(1:7)]
testing  <-testing[,-c(1:7)]

set.seed(200)
# Random subsampling without replacement (60%)
subsamples= sample(1:nrow(training),size=nrow(training)*0.6,replace=F)
subTraining <- training[subsamples, ] 
dim(subTraining)

subTesting <- training[-subsamples, ]
dim(subTesting)

#The goal is to predict the ???classe??? variable, so I examine this:
head(subTraining$classe)

# From the data source website, these classifications map to the following definitions:
#     
# Class A - exactly according to the specification
# Class B - throwing the elbows to the front
# Class C - lifting the dumbbell only halfway
# Class D - lowering the dumbbell only halfway
# Class E - throwing the hips to the front


#Frequency of levels (A, B, C, D, E) in the subTraining dataset for variable "classe"

summary(subTraining$classe)

qplot(subTraining$classe, 
      main="Levels of the variable classe")


#Correlation Analysis
correlation <- findCorrelation(cor(subTraining[, 1:ncol(subTraining)-1]), cutoff=0.8)
names(subTraining)[correlation]

#Prediction model 1: Decision Tree
modDecisionTree <- rpart(classe ~ ., 
                subTraining, 
                method="class")
modDecisionTree

prp(modDecisionTree,
    main="Decision Tree",
    box.col=rainbow(5))

plotcp(modDecisionTree)

tree_prune<-prune(modDecisionTree,cp=0.012)
prp(tree_prune,
    main="Decision Tree Prune",
    box.col=rainbow(5))

# prediction on Test dataset
predDecisionTree<-predict(modDecisionTree, subTesting,type ="class")


mDecisionTree<-confusionMatrix(predDecisionTree, subTesting$classe)
mDecisionTree$table
mDecisionTree$overall['Accuracy']

# Method 2: Generalized Boosted Model
tc <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modGBM  <- train(classe ~ ., 
                 data=subTraining, 
                 # A well known R package for fitting boosted decision trees is gbm.
                 method = "gbm",
                 trControl = tc, 
                 verbose = FALSE)

modGBM$finalModel

# prediction on Test dataset
predGBM <- predict(modGBM, subTesting)
mGBM <- confusionMatrix(predGBM, subTesting$classe)
mGBM$table
mGBM$overall['Accuracy']

# Method 3: Neural Network
#tc <- trainControl(method = "boot",number = 5)
garbage <- capture.output(
  mod_nnet<-train(classe ~ ., 
                  data=subTraining,
                  trControl = tc,
                  method = "nnet")
  )

# prediction on Test dataset
pred_nnet <- predict(mod_nnet, subTesting)
m_nnet <- confusionMatrix(pred_nnet, subTesting$classe)
m_nnet$table
m_nnet$overall['Accuracy']

# Method 4: svmLinear
modSVM  <- train(classe ~ ., 
                 data=subTraining, 
                 method = "svmLinear"
                  )

# prediction on Test dataset
predSVM <- predict(modSVM, subTesting)
mSVM <- confusionMatrix(predSVM, subTesting$classe)
mSVM$table
mSVM$overall['Accuracy']


# Method 5: k-NN
modKNN  <- train(classe ~ ., 
                 data=subTraining, 
                 method = "knn"
                  )



# prediction on Test dataset
predKNN <- predict(modKNN, subTesting)
mKNN <- confusionMatrix(predKNN, subTesting$classe)
mKNN$table
mKNN$overall['Accuracy']


#Accuracy for each model
acc<-data.frame(Model=c("Decision Tree", "gbm", "Neural Network","svm Linear", "k-NN" ),
                Accuracy=c(mDecisionTree$overall['Accuracy'],mGBM$overall['Accuracy'],m_nnet$overall['Accuracy'],mSVM$overall['Accuracy'],mKNN$overall['Accuracy']),
                stringsAsFactors=FALSE)
acc

#Predictions for the best model (Generalized Boosted Model)
predict(modGBM, testing)



#Appendix 
str(subTraining)
head(summary(subTraining))

