setwd('C://Users//bhave//Desktop//code//DataScience_Practice')
getwd()

iris_data <- read.csv("dataset/iris.csv",header = FALSE)
View(iris_data)

"Attribute Information:
  1. sepal length in cm
  2. sepal width in cm
  3. petal length in cm
  4. petal width in cm
  5. class: 
  -- Iris Setosa
  -- Iris Versicolour
  -- Iris Virginica"

"we dont have header so add header to our data frame"
col_headings <- c('sepal_length_cm','sepal_width_cm','petal_length_cm','petal_width_cm','iris_class')
names(iris_data) <- col_headings
View(iris_data)

"Check missing Values"
`is.na<-`(iris_data)
is.na(iris_data)
which(is.na(iris_data))

is.na(iris_data)
apply(is.na(iris_data), 2, which)

"checking outliers"
str(iris_data)

for (val in col_headings)
{
  if(str(iris_data.obs) == "num")
  {
    boxplot(iris_data$val)  
  }
}




