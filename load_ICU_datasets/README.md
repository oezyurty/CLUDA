Before you start, we suggest you to install [RICU](https://cran.r-project.org/web/packages/ricu/index.html) package first and setup the database for MIMIC-IV and AmsterdamUMCdb with your credentials. 

Then, you can run [feature_selection.R](feature_selection.R) script first. Note that you can add/remove some time series measurements according to your experimental setup. 

In the same R environment, you can finally run [process_miiv.R](process_miiv.R) and [process_aumc.R](process_aumc.R) for MIMIC-IV and AmsterdamUMCdb. 
