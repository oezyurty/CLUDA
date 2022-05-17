#Load some required libraries
library(ricu)
library(dplyr)
library(caret)
set.seed(1234)

# SECTION 1

#For all patients, we will use only the first admission to ICU.
#Get the admissions table from the database
admissions_aumc_df <- as.data.frame(aumc$admissions)

#Filter out all 2nd, 3rd etc. admissions
admissions_aumc_df <- admissions_aumc_df[admissions_aumc_df$admissioncount==1,]

#Only get the Intensive-care related admission (FOR NOW, NOT APPLIED)
#admissions_aumc_df <- admissions_aumc_df[admissions_aumc_df$location != "MC",]

#SANITY CHECK: calculate the length of stay and check with the given column. (PASSED)
#admissions_aumc_df$LOS <- admissions_aumc_df$dischargedat/1000/60/60
#mean(abs(admissions_aumc_df$LOS- admissions_aumc_df$lengthofstay))

#We will calculate the death flag. 

# If dateofdeath is negative, it refers to short admissions (usually <24 hours), we will discard them (268 stays discarded)
flag_neg_death <- !is.na(admissions_aumc_df$dateofdeath) & admissions_aumc_df$dateofdeath <0
admissions_aumc_df <- admissions_aumc_df[!flag_neg_death,] 

# If death time exists, and it's less than dischargetime, we will count it as ICU mortality

#According to Official Github, 1 day discrepancy between dateofdischarge and dateofdeath is possible. 
#We can count above case as ICU mortality safely.
day_milisec <- 1000*60*60*24
admissions_aumc_df$deathdiscdiff <- (admissions_aumc_df$dateofdeath - admissions_aumc_df$dischargedat)/day_milisec

#If dateofdeath - dischargedat < 1 day, we will accept those cases as in-hospital mortality. 
admissions_aumc_df$mortality_label <- NA
admissions_aumc_df$mortality_label[admissions_aumc_df$deathdiscdiff<1 & !is.na(admissions_aumc_df$deathdiscdiff)] <- 1
admissions_aumc_df$mortality_label[is.na(admissions_aumc_df$mortality_label)] <- 0


# SECTION 2

#Load cohort with static features (coming from load_mimic.R)
cohort_aumc <- load_concepts(static_features, "aumc", patient_ids=admissions_aumc_df$admissionid, verbose = FALSE)

#Load timeseries features (for selected timeseries)
ts_data_aumc <- load_concepts(selected_ts_features, "aumc", 
                              patient_ids = cohort_aumc, verbose = TRUE)


#remove timesteps that are longer than length of stay (might be redundant.)
for(aid in admissions_aumc_df$admissionid){
  los <- admissions_aumc_df$lengthofstay[admissions_aumc_df$admissionid == aid]
  cond <- ts_data_aumc$admissionid == aid & ts_data_aumc$measuredat > los
  ts_data_aumc <- ts_data_aumc[!cond,]
}

#Fill the missing time steps with full NAs
ts_data_aumc <- fill_gaps(ts_data_aumc)

#Get length of stays for each admission
adm_los_aumc <- ts_data_aumc %>% group_by(admissionid) %>% summarise(los=max(measuredat))

#Below histograms were just for observation
#hist(as.numeric(adm_los$los), breaks=100, freq=F, main="Distribution of LOS")
#hist(trim_q(as.numeric(adm_los$los), 0.05, 0.9), breaks=100, freq=F, main="Distribution of LOS after trimming")

#For forward fill, we won't use the measurements older than 48 hours(before ICU admission)
#So, we remove everything below -48 hours
ts_data_aumc <- ts_data_aumc[ts_data_aumc$measuredat > -48,]


#Before we do any forward filling etc. let's keep the mask of actual features (might be useful later)
ts_data_aumc_mask <- ts_data_aumc
ts_data_aumc_mask <- ts_data_aumc_mask[ts_data_aumc_mask$measuredat > -1]
for(i in 3:ncol(ts_data_aumc_mask)){
  col_name <- colnames(ts_data_aumc_mask)[i]
  ts_data_aumc_mask[[col_name]] <- !is.na(ts_data_aumc_mask[[col_name]])
}

ts_data_aumc_mask_numeric <- ts_data_aumc_mask
for(i in 3:ncol(ts_data_aumc_mask_numeric)){
  col_name <- colnames(ts_data_aumc_mask_numeric)[i]
  ts_data_aumc_mask_numeric[[col_name]] <- as.numeric(ts_data_aumc_mask_numeric[[col_name]])
}

ts_data_aumc_mask <- ts_data_aumc_mask_numeric
rm(ts_data_aumc_mask_numeric)

#SECTION 3
#In order to do imputation (with mean), we need training to prevent information leak to val-test set
train_index <- createDataPartition(admissions_aumc_df$mortality_label, p = .7, list = FALSE)

train_admissions <- admissions_aumc_df$admissionid[train_index]

#Get validation and test admissions here as well
val_index <- createDataPartition(admissions_aumc_df$mortality_label[-train_index], p = .5, list = FALSE)

val_admissions <- admissions_aumc_df$admissionid[-train_index][val_index]
test_admissions <- admissions_aumc_df$admissionid[-train_index][-val_index]

#Do forward filling (for all admissions)
ts_data_aumc <- replace_na(ts_data_aumc, rep(NA, length(selected_ts_features)), type = rep("locf", length(selected_ts_features)),
                      by_ref = TRUE, vars = selected_ts_features,
                      by = id_vars(ts_data_aumc))

#After filling, we don't need the measurements before ICU admission anymore
ts_data_aumc <- ts_data_aumc[ts_data_aumc$measuredat > -1]

#IMPORTANT: only one manual intervention to dataset
#There is one LARGE value for wbc at admission 15808. We will fix it manually
ts_data_aumc$wbc[ts_data_aumc$wbc > 1000000] <- 9.7 #which is the previous measurement of a patient

#Calculate mean and std based 'only' on training set
#dictionary of mean values (across all training admissions) for a given measurement
ts_aumc_means <- colMeans(ts_data_aumc[ts_data_aumc$admissionid %in% train_admissions,..selected_ts_features], na.rm = T)

#dictionary of standard devation values (across all training admissions) for a given measurement
ts_aumc_sds <- apply(ts_data_aumc[ts_data_aumc$admissionid %in% train_admissions,..selected_ts_features],2,sd, na.rm=T)

#SECTION 4: Apply scaling to all data
ts_data_aumc_scaled <- ts_data_aumc

#Do the scaling operation column-wise
for(i in 3:ncol(ts_data_aumc_scaled)){
  col_name <- colnames(ts_data_aumc_scaled)[i]
  ts_data_aumc_scaled[[col_name]] <- (ts_data_aumc_scaled[[col_name]] - ts_aumc_means[col_name])/ts_aumc_sds[col_name]
}

#Fill the na values with zeros
ts_data_aumc_scaled[is.na(ts_data_aumc_scaled)] <- 0

#In this process, we (might) have removed some admissions, so update the admissions_aumc_df as well
admissions_aumc_df <- admissions_aumc_df[admissions_aumc_df$admissionid %in% unique(ts_data_aumc_scaled$admissionid),]

#In admissions_table, keep the split type of each admission
admissions_aumc_df$splitType <- NA
admissions_aumc_df$splitType[admissions_aumc_df$admissionid %in% train_admissions] <- "train"
admissions_aumc_df$splitType[admissions_aumc_df$admissionid %in% val_admissions] <- "val"
admissions_aumc_df$splitType[admissions_aumc_df$admissionid %in% test_admissions] <- "test"

#SECTION 5: Save the current data (i.e. time-series and admission info)
dir.create("aumc")

write.csv(ts_data_aumc_scaled,"aumc/ts_data_scaled.csv", row.names = FALSE)
write.csv(ts_data_aumc_mask,"aumc/ts_data_mask.csv", row.names = FALSE)
write.csv(admissions_aumc_df,"aumc/admissions.csv", row.names = FALSE)


#SECTION 6: Finalize static features as well
#Note: static_features comes from load_mimic.R 

#subset cohort_aumc after (possibly) removing some admission ids
cohort_aumc <- cohort_aumc[cohort_aumc$admissionid %in% admissions_aumc_df$admissionid,]

#There is one categorical feature called adm: surg=0, med=1, {other,NA}=2
cohort_aumc$adm_categ <- NA
cohort_aumc$adm_categ[cohort_aumc$adm == "surg"] <- 0
cohort_aumc$adm_categ[cohort_aumc$adm == "med"] <- 1
cohort_aumc$adm_categ[is.na(cohort_aumc$adm_categ)] <- 2

#There is another categorical feature called sex: male=0, female=1
cohort_aumc$sex_categ <- NA
cohort_aumc$sex_categ[cohort_aumc$sex == "Female"] <- 1
cohort_aumc$sex_categ[is.na(cohort_aumc$sex_categ)] <- 0 #majority is male, that's why we fill NAs as 0

#Now apply standardization to continuous static features
static_cont_features <- setdiff(static_features, c("adm", "sex"))

#dictionary of mean values (across all training admissions) for a given measurement
static_aumc_means <- colMeans(cohort_aumc[cohort_aumc$admissionid %in% train_admissions,..static_cont_features], na.rm = T)

#dictionary of standard devation values (across all training admissions) for a given measurement
static_aumc_sds <- apply(cohort_aumc[cohort_aumc$admissionid %in% train_admissions,..static_cont_features],2,sd, na.rm=T)

cohort_aumc_scaled <- cohort_aumc[,-c("adm", "sex")]

#Do the scaling operation column-wise
for(col_name in static_cont_features){
  cohort_aumc_scaled[[col_name]] <- (cohort_aumc_scaled[[col_name]] - static_aumc_means[col_name])/static_aumc_sds[col_name]
}

#Fill the na values with zeros
cohort_aumc_scaled[is.na(cohort_aumc_scaled)] <- 0

#Save the static features as well
write.csv(cohort_aumc_scaled,"aumc/static_data_scaled.csv", row.names = FALSE)

