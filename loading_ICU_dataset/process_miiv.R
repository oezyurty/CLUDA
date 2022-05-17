#Load some required libraries
library(ricu)
library(dplyr)
library(caret)
set.seed(1234)

#SECTION 1

#Get the admissions table from the database
admissions_df <- as.data.frame(miiv$admissions)
icustays_df <- as.data.frame(miiv$icustays)
patients_df <- as.data.frame(miiv$patients)

#filter ICU stays that we use only the first stays
first_stay_time <- icustays_df %>% group_by(subject_id) %>% summarise(intime = min(intime))
icustays_df <- icustays_df %>% inner_join(first_stay_time, by=colnames(first_stay_time))

#We use icustays that have no transfer.
icustays_df <- icustays_df[icustays_df$first_careunit == icustays_df$last_careunit, ]


#Get Mortality label from admissions_df (with hospital_expire_flag)
icustays_df <- icustays_df %>% inner_join(admissions_df[,c("hadm_id", "hospital_expire_flag")], by="hadm_id")


#SECTION 2
#Load cohort with static features (coming from load_mimic.R)
cohort <- load_concepts(static_features, "miiv", patient_ids=icustays_df$stay_id, verbose = FALSE)

#Load timeseries features (for selected timeseries)
ts_data <- load_concepts(selected_ts_features, "miiv", 
                              patient_ids = cohort, verbose = TRUE)

#Remove timesteps before 48 hours to admission, and after length of stay
find_los <- function(stay_id){
  hadm_id <- icustays_df$hadm_id[icustays_df$stay_id == stay_id]
  los <- icustays_df$los[icustays_df$hadm_id == hadm_id]
  los <- ceiling(los*24)
  return(los)
}

count <- 0
list_stayids <- unique(ts_data$stay_id)
for(sid in list_stayids){
  hadm_id <- icustays_df$hadm_id[icustays_df$stay_id == sid]
  los <- icustays_df$los[icustays_df$hadm_id == hadm_id]
  los <- ceiling(los*24)
  cond <- ts_data$stay_id == sid & ts_data$charttime > los
  ts_data <- ts_data[!cond,]
  if(count %% 100 == 0){print(count)}
  count <- count + 1
}

ts_data_temp <- ts_data[ts_data$charttime>-48,]
ts_data <- ts_data_temp 
rm(ts_data_temp)
#ts_data <- ts_data %>% group_by(stay_id) %>% filter(between(charttime, -48, find_los(stay_id[1]))) %>% ungroup

#Fill the missing time steps with full NAs
ts_data <- fill_gaps(ts_data)

#Just a quick backup (Above timestep removal takes too much time.)
#ts_data_backup <- ts_data 

#Get length of stays for each admission
adm_los <- ts_data %>% group_by(stay_id) %>% summarise(los=max(charttime))

#Below histograms were just for observation
hist(as.numeric(adm_los$los), breaks=100, freq=F, main="Distribution of LOS")
hist(trim_q(as.numeric(adm_los$los), 0.05, 0.9), breaks=100, freq=F, main="Distribution of LOS after trimming")


#Before we do any forward filling etc. let's keep the mask of actual features (might be useful later)
ts_data_mask <- ts_data
ts_data_mask <- ts_data_mask[ts_data_mask$charttime > -1]
for(i in 3:ncol(ts_data_mask)){
  col_name <- colnames(ts_data_mask)[i]
  ts_data_mask[[col_name]] <- !is.na(ts_data_mask[[col_name]])
}

for(i in 3:ncol(ts_data_mask)){
  col_name <- colnames(ts_data_mask)[i]
  ts_data_mask[[col_name]] <- as.numeric(ts_data_mask[[col_name]])
}

#Some stay_id's in icustays_df don't exist in ts_data_mask (because they only have negative time steps)
#Filter icustays_df
icustays_df <- icustays_df[icustays_df$stay_id %in% unique(ts_data_mask$stay_id),]

#SECTION 3
#In order to do imputation (with mean), we need training to prevent information leak to val-test set
train_index <- createDataPartition(icustays_df$hospital_expire_flag, p = .7, list = FALSE)

train_admissions <- icustays_df$stay_id[train_index]

#Get validation and test admissions here as well
val_index <- createDataPartition(icustays_df$hospital_expire_flag[-train_index], p = .5, list = FALSE)

val_admissions <- icustays_df$stay_id[-train_index][val_index]
test_admissions <- icustays_df$stay_id[-train_index][-val_index]

#Do forward filling (for all admissions)
ts_data <- replace_na(ts_data, rep(NA, length(selected_ts_features)), type = rep("locf", length(selected_ts_features)),
                           by_ref = TRUE, vars = selected_ts_features,
                           by = id_vars(ts_data))

#After filling, we don't need the measurements before ICU admission anymore
ts_data <- ts_data[ts_data$charttime > -1]

#Calculate mean and std based 'only' on training set
#dictionary of mean values (across all training admissions) for a given measurement
ts_means <- colMeans(ts_data[ts_data$stay_id %in% train_admissions,..selected_ts_features], na.rm = T)

#dictionary of standard devation values (across all training admissions) for a given measurement
ts_sds <- apply(ts_data[ts_data$stay_id %in% train_admissions,..selected_ts_features],2,sd, na.rm=T)

#SECTION 4: Apply scaling to all data
ts_data_scaled <- ts_data

#Do the scaling operation column-wise
for(i in 3:ncol(ts_data_scaled)){
  col_name <- colnames(ts_data_scaled)[i]
  ts_data_scaled[[col_name]] <- (ts_data_scaled[[col_name]] - ts_means[col_name])/ts_sds[col_name]
}

#Fill the na values with zeros
ts_data_scaled[is.na(ts_data_scaled)] <- 0

#In icustays_df, keep the split type of each admission
icustays_df$splitType <- NA
icustays_df$splitType[icustays_df$stay_id %in% train_admissions] <- "train"
icustays_df$splitType[icustays_df$stay_id %in% val_admissions] <- "val"
icustays_df$splitType[icustays_df$stay_id %in% test_admissions] <- "test"

#SECTION 5: Save the current data (i.e. time-series and admission info)
dir.create("miiv")

write.csv(ts_data_scaled,"miiv/ts_data_scaled.csv", row.names = FALSE)
write.csv(ts_data_mask,"miiv/ts_data_mask.csv", row.names = FALSE)
write.csv(icustays_df,"miiv/admissions.csv", row.names = FALSE)

#SECTION 6: Finalize static features as well
#Note: static_features comes from load_mimic.R 

#subset cohort after (possibly) removing some admission ids
cohort <- cohort[cohort$stay_id %in% icustays_df$stay_id,]

#There is one categorical feature called adm: surg=0, med=1, {other,NA}=2
cohort$adm_categ <- NA
cohort$adm_categ[cohort$adm == "surg"] <- 0
cohort$adm_categ[cohort$adm == "med"] <- 1
cohort$adm_categ[is.na(cohort$adm_categ)] <- 2

#There is another categorical feature called sex: male=0, female=1
cohort$sex_categ <- NA
cohort$sex_categ[cohort$sex == "Female"] <- 1
cohort$sex_categ[is.na(cohort$sex_categ)] <- 0 #majority is male, that's why we fill NAs as 0