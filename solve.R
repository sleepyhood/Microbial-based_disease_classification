############################
## 2023
## 질병 예측 모델 머신러닝
## 질병 예측 모델 구축
## 최적의 특징 추출 
## 민감도와 특이도, 정확도를 바탕으로 질병예측 가능성 추측하기
############################

########공통 작업###########
# 1. 모든 패키지 다운로드
setRepositories(ind = 1:7)

# 2-1. 작업경로 설정(*단, 역슬래시 2번씩 표기)
WORK_DIR <- "C:\\Users\\osw\\Desktop\\#Workspace\\GIT_BackUp\\R\\Microbial-based_disease_classification"
setwd(WORK_DIR) # 작업디렉토리 설정
getwd() # 작업디렉토리 확인


# 2-2. 데이터 폴더 가져오기
DATA_DIR<-paste0(WORK_DIR, "\\GivenMicrobialData")
setwd(DATA_DIR)
getwd()




# 3. 라이브러리 불러오기
## 그래프 함수
library(ggplot2)
library(gganimate)

##tibble() 자료형
library(tidyverse)

##날짜 처리 
library('lubridate')

## clean_names(), tabyl()
library('janitor')

## 문자열 내부에 변수 값 {x} 삽입
library(glue)

# 데이터프레임 기반 라이브러리
library(data.table)#fread를 사용하기 위한 라이브러리

library(caret)
library(robotstxt)
library(rvest) # 웹 스크래퍼
library(RSelenium)
library(dplyr)
library(httr)
library(jsonlite)
library(readr)
#install.packages("fst")
library(fst)

#install.packages("feather")
library(feather)
library(stats)

# 영어로 설정해야 월을 Date 객체로 바꿀 때 NA를 반환하지 않는다
Sys.setlocale("LC_TIME", "en_US.UTF-8")


# 1. 파일읽기
# data1,2,3 변수명에 원본 데이터
data1 <- read_tsv("MicrobiomeData_Hospital1.tsv")
data2 <- read_feather("MicrobiomeData_Hospital2.feather")
data3 <- read_fst("MicrobiomeData_Hospital3.fst")

# 병합
combined_data <- rbind(data1, data2, data3)
combined_data$Disease<-as.factor(combined_data$Disease)
combined_data$Disease

x<-combined_data %>% select(-Disease)
y<-combined_data$Disease
str(y)
str(x)
# View(x)
# 중복된 행은 검출되지 않음
subset(combined_data, duplicated(combined_data))

# 0618 추가?
#library(Boruta)
#Q5_boruta_result <- Boruta(Disease ~., data=combined_data, doTrace=2, maxRuns=15)
#Q5_boruta_result
# filtered feature selection 적용하기

## 
# install.packages("FSelector")
library(FSelector)
filtered_features <- information.gain(Disease ~ ., data = combined_data)
dim(filtered_features)

filtered_df<-cbind(feature = colnames(x),attr_importance=filtered_features)

filtered_df <- filtered_df[order(filtered_df$attr_importance, decreasing = TRUE), ]
# 중요한 미생물 5개
head(filtered_df, 5)

#filtered_df
filtered_features_top100 <- head(filtered_df$feature, 100)
filtered_features_top100

# 추출된 상위 100개의 변수를 포함하는 데이터프레임 생성
filtered_x <- combined_data[, filtered_features_top100]
filtered_x


###########################################################################

# Step 1
# 가장 정확도가 높은 모델 찾기


setwd(paste0(WORK_DIR, "\\Q5 모델 백업")) # 모델 백업 위치 지정
getwd() # 백업 디렉토리 확인



trainControl <- trainControl(method = "cv", number = 10,verboseIter =TRUE)

df <- data.frame(Model = character(), Accuracy = numeric(), stringsAsFactors = FALSE)
time_df <- data.frame(Model = character(), Time = numeric(), stringsAsFactors = FALSE)

# 반복문으로 모델링 하는 부분 
# 시간이 좀 걸림
models <- c(
  "knn",
  # "wsrf",
  #"glmnet",
  "RSimca",
  "rpart1SE",
  "C5.0Rules",
  "LogitBoost",
  "ranger"
)

########현재 모델 사용시############

View(cbind(filtered_x, y))
for (model_name in models) {
  print(model_name)
  start_time <- Sys.time()
  # knn은 importance 인자 사용 안함
  if(model_name=="knn" || model_name=="rpart1SE"){
    model <- train(y ~ ., data = cbind(filtered_x, y), method = model_name, trControl = trainControl)
  }
  else{
    model <- train(y ~ ., data = cbind(filtered_x, y), method = model_name, trControl = trainControl,importance = "impurity")
  }
  
  # 모델 저장
  saveRDS(model, paste0("Q5_filtered_model_", model_name, ".RData"))
  
  # 모델 실행 시간 측정 종료
  end_time <- Sys.time()
  elapsed_time <- end_time - start_time
  
  
  accuracy <- model$results$Accuracy  # 정확도 추출
  df <- rbind(df, data.frame(Model = model_name, Accuracy = accuracy, stringsAsFactors = FALSE))
  
  time_df<-rbind(time_df, data.frame(Model = model_name, Time = elapsed_time, stringsAsFactors = FALSE))
}
time_df


########현재 모델 사용시############


# 각 모델마다 정확도를 묶어서 순위를 매긴다.
df2<-df %>%
  group_by(Model) %>%
  mutate(Avg_Accuracy = mean(Accuracy)) %>%
  ungroup() %>%
  arrange(desc(Avg_Accuracy)) %>%
  mutate(Rank = dense_rank(-Avg_Accuracy)) %>% 
  arrange(Rank)
df2
# 박스 플롯과 산점도 그래프를 같이 표기
df2 %>%
  ggplot(aes(x = factor(Model, levels = unique(Model)), y = Accuracy, fill = Model)) +
  geom_boxplot(fill = NA, aes(color = Model)) +
  geom_jitter(shape = 16, aes(color = Model), size = 6, alpha = 0.3) +
  xlab("Model") +
  ylab("Accuracy") +
  ggtitle("Accuracy by Model (Ranked)") +
  theme_minimal() +
  theme(legend.position = "bottom")


#########################################################################################

# Step 2
# 정확도가 높은 모델을 기준으로 feature 찾기
# ranger 모델을 기준으로 하였습니다.

rangerModel<-readRDS("Q5_filtered_model_ranger.rData")
rangerImp<-varImp(rangerModel)
rangerImp
ordered_features <- rownames(rangerImp$importance)[order(-rangerImp$importance$Overall, decreasing = FALSE)]
ordered_features

top_20_features <- head(ordered_features, 100)
top_20_features

varImp_x <- filtered_x[, top_20_features]
varImp_x
trainControl <- trainControl(method = "cv", number = 2,verboseIter =TRUE)
# 상대적으로 중요한 데이터라고 해도, 순차적인 접근은 항상 죄적값을 보장하지 않음
Q5_varImp_model_ranger <- train(y ~ ., data = cbind(varImp_x, y), method = "ranger", trControl = trainControl,importance = "impurity")
# 정확도들
(Q5_varImp_model_ranger$results$Accuracy)
# 평균 정확도
mean(Q5_varImp_model_ranger$results$Accuracy)

#########################################################################################

# Step 3
# 각 질병마다 정확도, 민감도, 특이도 계산
# ranger 모델을 기준으로 하였습니다.
# 비교를 위해, knn도 사용했습니다.
rangerModel<-readRDS("Q5_filtered_model_ranger.rData")
knnModel<-readRDS("Q5_filtered_model_knn.rData")

testData<-cbind(filtered_x, y)
dim(testData)
# ranger
ranger_filtered_pred <- predict(rangerModel, newdata = testData %>% select(-y))
ranger_filtered_pred_result <- confusionMatrix(ranger_filtered_pred, y)
ranger_filtered_pred_df<-ranger_filtered_pred_result$byClass

ranger_filtered_pred_df[, c("Sensitivity", "Specificity", "Balanced Accuracy")]

# knn 모델
knn_filtered_pred <- predict(knnModel, newdata = testData %>% select(-y))
knn_filtered_pred_result <- confusionMatrix(knn_filtered_pred, y)
knn_filtered_pred_df<-knn_filtered_pred_result$byClass

knn_filtered_pred_df[, c("Sensitivity", "Specificity", "Balanced Accuracy")]

# 모델의 행을 랜덤하게 배치 후 결과
# shuffled_df <- filtered_dF[sample(nrow(filtered_df)), ]
# shuffled_pred <- predict(rangerModel, newdata = shuffled_df)
# shuffled_pred_result <- confusionMatrix(shuffled_pred, y)
# shuffled_pred_df<- shuffled_pred_result$byClass
# (shuffled_pred_df[, c("Sensitivity", "Specificity", "Balanced Accuracy")])


######################