import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.SparkSession

object Main {
    def main(args: Array[String]) = {
        val dataPath = "NewFiat.csv"

        val spark = SparkSession.builder
            .appName("Car Price Regression")
            .config("spark.master", "local")
            .getOrCreate()

        val df = spark.read
            .option("header", true)
            .csv(dataPath)

        val columnsToIndex = Seq("Brand", "Model", "Production Year", "Fuel Type", "Transmission Type", "Class", "Color", "Country Of Origin",
            "Registered In Poland", "First Owner", "Without Accidents", "Certified Auto Repair Serviced", "Usage State",
            "Currency")

        //val columnsToIndexWithDistinctRows = columnsToIndex.filter(column => df.select(column).distinct().count() > 1)

        val indexAppendix  = " Index"
        val vectorAppendix = " Vector"

        val Array(trainingData, testData) = df.randomSplit(Array(0.8, 0.2))

        val indexers = columnsToIndex.map(column =>
            new StringIndexer().setInputCol(column).setOutputCol(column + indexAppendix))

        /*
        val encoders = columnsToIndex.map(column =>
            new OneHotEncoder().setInputCol(column + indexAppendix).setOutputCol(column + vectorAppendix))
            .toArray
        */

        val featureColumns = (df.columns.toSet -- columnsToIndex - "Price" ++ columnsToIndex.map(_ + indexAppendix)).toSeq

        val assembler = new VectorAssembler()
            .setInputCols(featureColumns.toArray)
            .setOutputCol("features")

        val pipeline = new Pipeline().setStages((indexers).toArray)

        val processed = pipeline.fit(df).transform(df).select("Price", featureColumns: _*)

        processed.show()

        processed.printSchema()

        spark.stop()
    }
}