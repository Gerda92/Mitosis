library("EBImage")
library("Matrix")

image <- readImage('E:/DataMLMI/Slice1/30 Gy 2 Wo Le 2 65_part1.png')
mask <- readImage('E:/DataMLMI/GTSlice1/Labels_30 Gy 2 Wo Le 2 65_part1.png')

mf <- medianFilter(image[400:800, 900:1300,], 5, cacheSize=512)

display(mf)

display(image)
display(mask)

p1 = 420:600
p2 = 970:1120
patch(p1, p2)

p1 = 1380:1580
p2 = 1470:1660
patch(p1, p2)

p1 = 1950:2050
p2 = 1870:1970
patch(p1, p2)


p1 = 1:4300
p2 = 1:4236
patch(p1, p2)

p1 = 4301:6660
p2 = 1:4236
patch(p1, p2)

4300/6660

patch <- function (p1, p2) {
  display(image[p1, p2,])
  display(mask[p1, p2])
  nnzero(mask[p1, p2])/dim(mask[p1, p2])[1]/dim(mask[p1, p2])[2]
}


data <- data.frame(r = as.vector(image[400:600, 900:1100,1]),
                   g = as.vector(image[400:600, 900:1100,2]),
                   b = as.vector(image[400:600, 900:1100,3]),
                   class=as.factor(as.vector(mask[400:600, 900:1100])))


library(kernlab)
rbf <- rbfdot(sigma=0.05)
svm <- ksvm(data$class~., data=data, type="C-bsvc", kernel=rbf, C=10)

prediction <- predict(svm, data)
table(pred=prediction, true=data$class)

