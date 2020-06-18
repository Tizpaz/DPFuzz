# This also works :)
library('fda')
library('fda.usc')
library('dbscan')
library("Cairo")
library("conclust")
library("data.table")

args = commandArgs(trailingOnly=TRUE)
# args[1] --> name of input file
# args[2] --> # number of clusters
# args[3] --> # maximum number of tolerance for disturbance
# args[4] --> # whether write label to an output file
# args[5] --> # = 1 based on the slope, = 2 based on distance (coefficient of determinations)
              # = 3 based on l-0 norm

# args <- vector()
# args[1] <- "examples/decision_tree_final_time_analysis.csv"
# args[2] <- 1               # number of clusters
# args[3] <- 3              # maximum number of tolerance for disturbance
# args[4] <- 1               # whether write label to an output file
# args[5] <- 1               # = 1 based on the slope, = 2 based on distance (coefficient of determinations)
#                            # = 3 based on l-0 norm

num_clusters <- as.numeric(args[2])

MyData <- as.matrix(read.csv(file=args[1]))
rows <- dim(MyData)[1]
cols <- dim(MyData)[2]
distinct_labels <- length(unique(MyData[,3]))
rows_per_label <- nrow(subset(MyData, MyData[,3]=="0"))
Data_time <- matrix(nrow=rows_per_label,ncol=distinct_labels)
Data_size <- matrix(nrow=rows_per_label,ncol=distinct_labels)
index <- 0
label <- vector(length = distinct_labels)
for(j in 1:distinct_labels)
{
  res <- 0
  for (r in 1:rows_per_label)
  {
    results <- subset(MyData, MyData[,3]==as.character(j-1))[r,4]
    if(results > res)
      res = 1
  }
  if(res == 0){      # Program returns FALSE
    label[j] = num_clusters+1
    next;
  }

  time_data = vector(length = rows_per_label)
  time_data = subset(MyData, MyData[,3]==as.character(j-1))[,2]
  cnt <- 0
  for(i in 1:length(time_data))
  {
    if(i >= length(time_data)){
      break
    }
    if(time_data[i] > time_data[i+1]){
      cnt <- cnt + 1
    }
  }
  if(cnt <= args[3])
  {
    index <- index + 1
    Data_time[,index] <- subset(MyData, MyData[,3]==as.character(j-1))[,2]
    Data_size[,index] <- subset(MyData, MyData[,3]==as.character(j-1))[,1]
  }
  else
  {
    label[j] = num_clusters+2
  }
}
Data_time <- Data_time[,1:index]
Data_size <- Data_size[,1:index]
X <- rows_per_label
N <- index

namefile = strsplit(args[1],"/")[[1]][2]

init = paste("Figures/initial_",namefile, ".png",sep = '')
Cairo(file=init,
      bg="white",
      type="png",
      units="in",
      width=12,
      height=9,
      pointsize=12,
      dpi=200)
par(mar=c(7, 6, 1, 1) + 1.0)

matplot(Data_size,Data_time/1000000,type='l', axes=F, xlab = '', ylab = '', lwd = 2, xlim=range(Data_size))
axis(2, ylim=range(Data_time+10),lwd=2, cex.axis = 3, font = 2)
mtext(2,text="Time (s)",line=4, cex = 3)
axis(1, xlim=range(Data_size),lwd=2, cex.axis = 3, font = 2)
mtext(1,text="Size (features * samples)",line=4, cex = 3)
dev.off()

start_time <- as.numeric(Sys.time())*1000

lin_models <- list()
coef_models = list()
linear_or_exp = vector()    # 1 for linear 2 for exp
max_val = vector()
# fit linear functions
for(col in 1:index)
{
  fit1  <- lm(Data_time[,col]~Data_size[,col])
  fit2 <-  lm(log(Data_time[,col])~log(Data_size[,col]))
  if(fit1$residuals < fit2$residuals){
    lin_models[[col]] <- fit1$fitted.values
    # coef_models[[col]] <- c(fit1$coefficients[1],fit1$coefficients[2],0)
    coef_models[[col]] <- c(fit1$coefficients[2])
    linear_or_exp[col] <- 0
    max_val[col] <- max(fit1$fitted.values)
  }else{
    lin_models[[col]] <- exp(fit2$fitted.values)
    # coef_models[[col]] <- c(exp(fit2$coefficients[1]),fit2$coefficients[2],1)
    coef_models[[col]] <- c(fit1$coefficients[2])
    linear_or_exp[col] <- 1
    max_val[col] <- max(exp(fit2$fitted.values))
  }
}
type_of_clustering <- args[5]
if(type_of_clustering == 1)
{
  dist <- matrix(, nrow = index, ncol = 1)
  for(i in 1:index)
  {
    dist[i,] <- coef_models[[i]]
  }
  dist[dist<0.001] <- 0
  km <- kmeans(dist,num_clusters)
  clusterCut <- km$cluster
}else if(type_of_clustering == 2){
  dist <- matrix(, nrow = index, ncol = index)
  for(i in 1:index)
  {
    for(j in 1:index)
    {
      dist[i,j] <- abs(cor(lin_models[[i]],lin_models[[j]], method="pearson"))
      dist[i,j] <- ((1/2^dist[i,j])-0.5)*2
    }
  }
  dist[dist<0.08] <- 0.0
  print(dist)
  km <- hclust(dist(dist))
  clusterCut <- as.vector(cutree(km, num_clusters))
}else{
  km <- kmeans(max_val,num_clusters)
  clusterCut <- km$cluster
}


index <- 1
for(i in 1:length(label))
{
  if(label[i] <= num_clusters)
  {
    label[i] <- clusterCut[index]
    index <- index + 1
  }
}

end_time <- as.numeric(Sys.time())*1000

# Do the following for better plots!
newclusterCut <- clusterCut
# newclusterCut <- replace(newclusterCut, newclusterCut==2, 6)
# newclusterCut <- replace(newclusterCut, newclusterCut==4, 2)
# newclusterCut <- replace(newclusterCut, newclusterCut==6, 4)
# newclusterCut <- replace(newclusterCut, newclusterCut==3, 6)
# newclusterCut <- replace(newclusterCut, newclusterCut==4, 3)
# newclusterCut <- replace(newclusterCut, newclusterCut==6, 4)


# colors are encoded in palette() function
# "black"   "red"     "green3"  "blue"    "cyan"    "magenta" "yellow"  "gray"
final = paste("Figures/final_",namefile, ".png",sep = '')
Cairo(file=final,
      bg="white",
      type="png",
      units="in",
      width=12,
      height=9,
      pointsize=14,
      dpi=200)
par(mar=c(7, 6, 2,2) + 0.2)

matplot(Data_size,Data_time/1000000,type='l', axes=F, xlab = '', ylab = '', lwd = 2, xlim=range(Data_size),col=newclusterCut)
axis(2, ylim=range(Data_time),lwd=2, cex.axis = 3, font = 2)
mtext(2,text="Time (s)",line=4, cex = 3)
axis(1, xlim=range(Data_size),lwd=2, cex.axis = 3, font = 2)
mtext(1,text="Size (features * samples)",line=4, cex = 3)
dev.off()

if(args[4]==1){
  csv_file = paste("labels/label_", namefile , ".csv",sep = '')
  write.table(label, file = csv_file, append = FALSE, quote = TRUE, sep = ",",
              eol = "\n", na = "NA", dec = ".", row.names = TRUE,
              col.names = TRUE, qmethod = c("escape", "double"),
              fileEncoding = "")
  new_label <- rep(label,each=rows_per_label)
  csv_file = paste("labels/label_multiple_times_", namefile , ".csv",sep = '')
  write.table(new_label, file = csv_file, append = FALSE, quote = TRUE, sep = ",",
              eol = "\n", na = "NA", dec = ".", row.names = TRUE,
              col.names = TRUE, qmethod = c("escape", "double"),
              fileEncoding = "")  
}

print("execution time: ")
print(end_time-start_time, digits=15)
print("The figire is generated in")
print(final)