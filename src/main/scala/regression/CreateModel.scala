package regression

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.sql.DataFrame

import scala.util.Try

object CreateModel {
    def generateModel(dataFrame: DataFrame, columnNamesToIndex: Seq[String], labelColumnName: String) = {
        val indexAppendix  = " Index"

        val indexers = columnNamesToIndex.map(columnName =>
            new StringIndexer().setInputCol(columnName).setOutputCol(columnName + indexAppendix))

        val featureColumns = (dataFrame.columns.toSet -- columnNamesToIndex - labelColumnName ++ columnNamesToIndex.map(_ + indexAppendix)).toSeq

        val assembler = new VectorAssembler()
            .setInputCols(featureColumns.toArray)
            .setOutputCol("features")

        val gbt = new GBTRegressor()
            .setLabelCol(labelColumnName)
            .setFeaturesCol("features")
            .setPredictionCol("Predicted " + labelColumnName)
            .setMaxIter(gbtIterations)

        val pipeline = new Pipeline().setStages((indexers :+ assembler :+ gbt).toArray)

        Try(pipeline.fit(dataFrame))
    }

    def loadFromFile(modelFilename: String) = Try(PipelineModel.load(modelFilename)).toOption

    private val gbtIterations = 200
}
