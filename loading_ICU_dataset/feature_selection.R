#In this script, we aim to explore the features for both MIMIC and AUMC

library(ricu)
library(dplyr)

#Get the concepts (i.e. relevant features for our experiments)
table_concepts <- explain_dictionary()

static_features <- table_concepts$name[table_concepts$category == "demographics"]
icu_labels <- table_concepts$name[table_concepts$category == "outcome"]

timeseries_categories <- c("chemistry", "blood gas", "vitals", "hematology")
timeseries_features <- table_concepts$name[table_concepts$category %in% timeseries_categories]

#AUMC looks more suspicious, let's start with it
cohort_aumc <- load_concepts(static_features, "aumc", verbose = FALSE)

#Take first 2000 admissions to make data-loading etc. faster
#cohort_cond_aumc <- cohort_aumc[admissionid < 2000, ]

#Check the timeseries features of aumc (for full aumc now)
ts_data_aumc <- load_concepts(c(timeseries_features, "death"), "aumc", 
                              patient_ids = cohort_aumc, verbose = TRUE)

#Below vector tells us what percentage of ts_data is filled 
# Note that (current) ts_data only shows the time-steps(hours) that have at least one measurement.
filled_ratio_ts_aumc <- c()

#Below function is just a helper for removing outliers
trim_q <- function(x, lb, ub){
  x[(x > quantile(x, lb, na.rm=T)) & (x < quantile(x, ub, na.rm=T))]
}

# Now iterate over the features to see their missingness rate and their distribution
for(f in timeseries_features){
  print(paste("Exploring the feature:", f))
  filled_ratio <- mean(!is.na(ts_data_aumc[[f]]))
  print(paste("filled ratio:", filled_ratio))
  filled_ratio_ts_aumc <- c(filled_ratio_ts_aumc, filled_ratio)
  if(filled_ratio > 0){
    par(mfrow=c(2,1))
    hist(ts_data_aumc[[f]], breaks=100, freq=F, main=paste("Distribution of", f))
    hist(trim_q(ts_data_aumc[[f]], 0.01, 0.99), breaks=100, freq=F, main=paste("Distribution of", f, "after trimming"))
  }
  
  invisible(readline(prompt="Press [enter] to continue"))
}
names(filled_ratio_ts_aumc) <- timeseries_features

#Let's check the length of ICU stays of patients 
icu_lengths_aumc <- ts_data_aumc %>% group_by(admissionid) %>% summarise(max_hour = max(measuredat))
hist(icu_lengths_aumc$max_hour, breaks=100, freq=F, main="Distribution of icu lengths")




#Let's do similart anaylsis for MIMIC-4 to understand the filled ratio of features
cohort <- load_concepts(static_features, "miiv", verbose = FALSE)

#Check the timeseries features of aumc (for full miiv now)
ts_data <- load_concepts(c(timeseries_features, "death"), "miiv", 
                              patient_ids = cohort, verbose = TRUE)

#Below vector tells us what percentage of ts_data is filled for miiv
# Note that (current) ts_data only shows the time-steps(hours) that have at least one measurement.
filled_ratio_ts <- c()

# Now iterate over the features to see their missingness rate and their distribution
for(f in timeseries_features){
  print(paste("Exploring the feature:", f))
  filled_ratio <- mean(!is.na(ts_data[[f]]))
  print(paste("filled ratio:", filled_ratio))
  filled_ratio_ts <- c(filled_ratio_ts, filled_ratio)
  if(filled_ratio > 0){
    par(mfrow=c(2,1))
    hist(ts_data[[f]], breaks=100, freq=F, main=paste("Distribution of", f))
    hist(trim_q(ts_data[[f]], 0.01, 0.99), breaks=100, freq=F, main=paste("Distribution of", f, "after trimming"))
  }
  
  invisible(readline(prompt="Press [enter] to continue"))
}
names(filled_ratio_ts) <- timeseries_features

#Let's select feature that have +1% prevalence in both dataset
selected_ts_features <- intersect(names(filled_ratio_ts)[filled_ratio_ts > 0.01], names(filled_ratio_ts_aumc)[filled_ratio_ts_aumc > 0.01])
#Additionally we will add some features that exist +1% in MIMIC, and slightly less than 1% in AUMC
selected_ts_features <- c(selected_ts_features, "basos", "eos", "lymph", "neut", "pt", "rbc")
