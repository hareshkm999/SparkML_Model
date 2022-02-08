from pyspark.sql import SparkSession
from pyspark.sql.functions import  col, isnull, when, count
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession \
    .builder \
    .appName('Titanic Data') \
    .getOrCreate()

df = (spark.read
          .format("csv")
          .option('header', 'true')
          .load("train.csv"))

print("printng counts harish {} ".format(df.count()))
df.columns
print(df.describe().toPandas())

print(df.dtypes)

dataset = df.select(col('Survived').cast('float'),
                         col('Pclass').cast('float'),
                         col('Sex'),
                         col('Age').cast('float'),
                         col('Fare').cast('float'),
                         col('Embarked')
                        )

print(dataset.dtypes)

dataset.select([count(when(isnull(c), c)).alias(c) for c in dataset.columns]).show()

# Drop missing values
dataset = dataset.replace('null', None)\
        .dropna(how='any')

# Index categorical columns with StringIndexer

dataset = StringIndexer(
    inputCol='Sex',
    outputCol='Gender',
    handleInvalid='keep').fit(dataset).transform(dataset)
dataset = StringIndexer(
    inputCol='Embarked',
    outputCol='Boarded',
    handleInvalid='keep').fit(dataset).transform(dataset)
dataset.show()

# Drop unnecessary columns
dataset = dataset.drop('Sex')
dataset = dataset.drop('Embarked')

dataset.show()

required_features = ['Pclass',
                    'Age',
                    'Fare',
                    'Gender',
                    'Boarded'
                   ]

#from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=required_features, outputCol='features')

transformed_data = assembler.transform(dataset)

transformed_data.show()

# Split the data
(training_data, test_data) = transformed_data.randomSplit([0.8,0.2])

# Define the model
rf = RandomForestClassifier(labelCol='Survived',
                            featuresCol='features',
                            maxDepth=5)

# Fit the model
model = rf.fit(training_data)

# Predict with the test dataset
predictions = model.transform(test_data)

# Evaluate our model


evaluator = MulticlassClassificationEvaluator(
    labelCol='Survived',
    predictionCol='prediction',
    metricName='accuracy')

# Accuracy
accuracy = evaluator.evaluate(predictions)
print('Test Accuracy = ', accuracy)

