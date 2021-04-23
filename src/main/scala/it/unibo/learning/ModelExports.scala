package it.unibo.learning

import it.unibo.learning.RegressionImplicits.RichRegression
import smile.data.DataFrame
import smile.regression._

import scala.util.Try

object ModelExports extends App {
  val baseFolder = "src/main/resources/"

  def evalAndStore(name: String, model: => DataFrameRegression, input: DataFrame): Try[Unit] = {
    println(s"$name MSE = ${model.mse(input)}")
    model.store(s"$baseFolder$name")
  }

  val input = Dataset.load("output.csv")

  val regressors: Map[String, DataFrameRegression] = Map(
    "linear"         -> lm(Dataset.formula, input),
    "ridge"          -> ridge(Dataset.formula, input, 0.0001),
    "lasso"          -> lasso(Dataset.formula, input, 0.001),
    "cart"           -> cart(Dataset.formula, input),
    "random_forest"  -> randomForest(Dataset.formula, input),
    "gradient_boost" -> gbm(Dataset.formula, input)
  )

  regressors.foreach { case (name, regression) =>
    evalAndStore(name, regression, input)
  }
}
