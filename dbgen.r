options(echo=FALSE) # if you want see commands in output file
options(max.print=100000)
usePackage <- function(p) {
  if (!is.element(p, installed.packages()[,1]))
    install.packages(p, dep = TRUE)
  require(p, character.only = TRUE)
}

# Make a function to fill in the missing DXCHANGE for NA values
replacebyneighbour <- function(x){
  # For each NA value in this column replace it with its previous neighbour
  x[-length(x)] = na.locf(x, fromLast=TRUE)
  x = na.locf(x)
  return(x)
}

# Install ADNIMERGE library to use this script
# ADNIMERGE contains a dxsum table which has the DXCHANGE column filled for all subjects 
# Pre-requisite code:
#'''
usePackage("Hmisc") 
usePackage("R.matlab")
usePackage("zoo")

if (!is.element("ADNIMERGE", installed.packages()[,1]))
  install.packages("/home/siqi/Downloads/ADNIMERGE_0.0.1.tar.gz", repo=NULL, type="source")
require("ADNIMERGE", character.only = TRUE)

args <- commandArgs(trailingOnly = TRUE)
dbpath = args[1]
csvpath = args[2]
#dbpath = '/home/siqi/workspace/AdniDemonsDatabase/Testing/5ADNI-Patients/'
#csvpath = '/home/siqi/workspace/AdniDemonsDatabase/Testing/5ADNI-Patients/db.csv'

dbcsv = read.csv(csvpath)
sub_adnimerge = subset(adnimerge, select=c("RID", "PTID", "VISCODE"));
sub_dxsum = subset(dxsum, select=c("RID", "VISCODE", "DXCHANGE"))
# add ptid to dxsum
dxsum_ptid = merge(dxsum, sub_adnimerge, by.x=c("RID", "VISCODE"), by.y=c("RID", "VISCODE"))

# Make MRI meta infomation table to convert find the visit code of the mri images
mrimetafields = c("RID","VISCODE","EXAMDATE")
submri15meta = subset(mrimeta, select=mrimetafields) # 1.5T
submrigometa = subset(mri3gometa, select=mrimetafields)
submri3meta  = subset(mri3meta, select=mrimetafields) 
submrimeta = rbind(submri15meta, submrigometa, submri3meta)

# Get PTID for mri meta
#submridx <- merge(submrimeta, dxsum_ptid, by.x=c("RID", "VISCODE"), by.y=c("RID", "VISCODE"))

# Get VISCODE for dbcsv
sbjid <- data.frame(do.call('rbind', strsplit(as.character(dbcsv$Subject),'_S_',fixed=TRUE)))
dbcsv$siteid = as.integer(as.character(sbjid$X1))
dbcsv$RID = as.integer(as.character(sbjid$X2))
dbcsv$Acq.Date = as.Date(dbcsv$Acq.Date, "%m/%d/%Y")
dbcsv = merge(submrimeta, dbcsv,by.x=c("RID", "EXAMDATE"), by.y=c("RID", "Acq.Date"))
dbcsv$VISCODE[dbcsv$VISCODE=="scmri"]="sc"
dbcsv <- merge(dbcsv, sub_dxsum, by.x=c("RID", "VISCODE"), by.y=c("RID", "VISCODE"), all.x=TRUE)
dbcsv$VISCODE[dbcsv$VISCODE %in% c("scmri", "sc", "bl", "uns1")] = "m0"
dbcsv <- dbcsv[order(dbcsv$RID, dbcsv$VISCODE),]
#dbcsv$DXCHANGE = replacebyneighbour(dbcsv$DXCHANGE)

# Filter out the subjects with missing diagnosis
dbcsv = dbcsv[is.na(dbcsv$DXCHANGE)==FALSE, ]

# Filter out the subjects with only one VISIT
visfreq = aggregate(dbcsv, by=list(dbcsv$RID), FUN=function(x) length(unique(x)))
validFreqRID = visfreq[visfreq$VISCODE > 1, "Group.1"]
dbcsv = dbcsv[dbcsv$RID %in% validFreqRID,]

# Keep only the first scan if there are more than one scans available in one VISIT
visfreq = aggregate(dbcsv[,c("RID","VISCODE","Image.Data.ID")], by=list(dbcsv$RID, dbcsv$VISCODE), FUN=max)
dbcsv = dbcsv[dbcsv$Image.Data.ID %in% visfreq$Image.Data.ID, ]

write.table(dbcsv, file.path(dbpath, 'dbgen.csv'), sep=',', col.names=NA)
print(sprintf("dbgen.csv was generated in: %s", file.path(dbpath, 'dbgen.csv')))