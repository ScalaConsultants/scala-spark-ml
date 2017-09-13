import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.functions._
import regression.CreateModel

import scala.util.{Failure, Success}

object Main {
    def main(args: Array[String]) = {
        val dataPath = "UsedVolkswagen.csv"

        val spark = SparkSession.builder
            .appName("Car Price Regression")
            .config("spark.master", "local")
            .getOrCreate()

        spark.sparkContext.setLogLevel("ERROR")

        val df = spark.read
            .option("header", true)
            .csv(dataPath)

        val labelColumn = "Price"

        val columnsToIndex = Seq("Brand", "Model", "Production Year", "Fuel Type", "Transmission Type", "Class", "Color", "Country Of Origin",
            "Registered In Poland", "First Owner", "Without Accidents", "Certified Auto Repair Serviced", "Usage State",
            "Currency")

        val columnsToCast   = df.columns.toSet -- columnsToIndex
        val frameAfterCasts = columnsToCast.foldLeft(df)((frame, column) => frame.withColumn(column, df(column).cast(DoubleType)))

        val modelOption = CreateModel.generateModel(frameAfterCasts, columnsToIndex, labelColumn)

        //This will evaluate the same dataset as the model was fitted on, thus may be prone to overfitting bias
        val evaluator = new RegressionEvaluator()
            .setLabelCol(labelColumn)
            .setPredictionCol("Predicted " + labelColumn)
            .setMetricName("rmse")

        modelOption map (_.transform(frameAfterCasts).select("Predicted " + labelColumn, labelColumn, "features")) match {
            case Success(predictions) => {
                predictions.show()
                println("RMSE: " + evaluator.evaluate(predictions))
                frameAfterCasts.select(labelColumn).agg(avg(labelColumn).as("Average " + labelColumn)).show()
            }
            case Failure(error) => println("Failed! Error: " + error.getMessage)
        }

        spark.stop()
    }
}
