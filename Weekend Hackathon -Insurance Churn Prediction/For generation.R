#For generation
setwd("C:/Users/PR20046708/Desktop/python wipro assignments/Insurance_Churn_ParticipantsData/Insurance_Churn_ParticipantsData")
train = read.csv("Train.csv")
test = read.csv("Test.csv")

class(train)
summary(train)
str(train)

train$labels = as.factor(train$labels)


str(train)
table(train$labels)
3967/29941


anyNA(train)
anyNA(test)


#####################################################
#outlier detection and Treatment - Multivariate method
#####################################################

str(train)


#Multivariate Method - Cooks Distance

#take any independent variable and build a linar model
mod = lm(feature_0~., data = train)

#Store cooks distane of the model
cooksd = cooks.distance(mod)

#Plot cooks distance
plot(cooksd, pch="*", cex=1, main="Influential Obs by Cooks distance")  # plot cook's distance
abline(h = 4*mean(cooksd, na.rm=T), col="red")  # add cutoff line
text(x=1:length(cooksd)+1, y=cooksd, labels=ifelse(cooksd>4*mean(cooksd, na.rm=T),names(cooksd),""), col="blue", cex = 0.8)

# Find Outliers Values index positions
influential <- as.numeric(names(cooksd)[(cooksd > 4*mean(cooksd, na.rm=T))])

# Filter data for Outliers
out <- (train[influential, ]) 

#View(out) # View Outliers
str(out)
dim(out)
# Finding Data except outliers
# Using anti_join from dplyr

out1 <- anti_join(train, out)
str(out1)

#Compare variable by variable the means of outliers dataset with dataset without outliers
summary(out)
summary(out1)
# Only daymins variable does not have a change in mean value in outlier dataset and dataset without outliers

#Replace the values in the variable from outliers data set, 
#with mean of the values in variable from dataset without outliers

out$feature_0 <- mean(out1$feature_0)
out$feature_1 <- mean(out1$feature_1)
out$feature_2 <- mean(out1$feature_2)
out$feature_3 <- mean(out1$feature_3)
out$feature_4 <- mean(out1$feature_4)
out$feature_5 <- mean(out1$feature_5)
out$feature_6 <- mean(out1$feature_6)
out$feature_7 <- mean(out1$feature_7)
out$feature_14 <- mean(out1$feature_14)

final_data <- rbind(out1, out)

write.csv(final_data,"outlier treated data.csv")
str(final_data)

dim(final_data)

matrix = cor(final_data)
corrplot(matrix,method = "number", type = "upper")


str(final_data)


#####################################################
# SMOTE - Data Preparation
#####################################################

# Target variable


library(DMwR)
library(dplyr)
library(caTools)

table(final_data$labels)
3967/29941

prop.table(table(final_data$labels))

levels(final_data$labels)
table(final_data$labels)

balanced.data <- SMOTE(labels ~., final_data, perc.over = 650, k = 5, perc.under = 126)
#in SMOTE we have to define our equation
#perc.over means that 1 minority class will be added for every value of perc.over
table(balanced.data$labels)
prop.table(table(balanced.data$labels))
#3967y = 29941
7934/29911

#####################################################
#XGBoost
#####################################################
names(balanced.data)
str(balanced.data)
View(balanced.data)
write.csv(balanced.data,"balanced data.csv")

smote.train = balanced.data
smote.test = test

features_train <- as.matrix(smote.train[,-c(17)])
label_train <- as.matrix(smote.train[,17])
features_test <- as.matrix(smote.test[,-c(17)])



xgb.fit <- xgboost(data = features_train,
                   label = label_train,
                   eta = 0.001,
                   max_depth = 5,
                   min_child_weight = 5,
                   nrounds = 10000,
                   nfold = 10,
                   objective = "binary:logistic",
                   verbose = 0,
                   early_stopping_rounds = 1000)

labelstr <- predict(xgb.fit, newdata = features_train)
y_pred_num = ifelse(labelstr > 0.5,1,0)
predicted.tr= factor(y_pred_num,levels = c(0,1))
expected.tr= smote.train$labels
results.tr <- confusionMatrix(data=predicted.tr, reference=expected.tr)
results.tr

labelsts <- predict(xgb.fit, newdata = features_test)
y_pred_num = ifelse(labelsts > 0.5,1,0)
predicted.ts= factor(y_pred_num,levels = c(0,1))

predicted.ts

library(xlsx)
write.xlsx(predicted.ts, file="submission.xlsx",sheetName="Sheet1",col.names = TRUE, row.names = FALSE, append = FALSE)
predicted.ts

levels(predicted.ts)
table(predicted.ts)
