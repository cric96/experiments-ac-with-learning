package it.unibo.learning
import smile.data.Tuple
import smile.data.`type`.{DataTypes, StructField, StructType}
import smile.data.formula._
import smile.{read, write}
import smile.regression._

import scala.language.postfixOps

object RidgeModel extends App {
  import RegressionImplicits._

  val input = Dataset.load("output.csv")

  val model = ridge(Dataset.formula, input, 0.0001)

  def createTest(in: Double): Tuple = Dataset.createTuple(0, in)

  def createSource(in: Double): Tuple = Dataset.createTuple(1.0, in)
  write.xstream(model, "src/main/resources/model")
  //it learns min + 1:
  println(model.predict(createTest(1.0)))
  println(model.predict(createTest(100.0)))
  //the error with high number is consistent..
  println(model.predict(createTest(1000.0)))
  //it doesn't learn target => 0.0 (it learns min - 1)
  println(model.predict(createSource(0.0)))
  println(model.predict(createSource(100.0)))

  println("MSE = " + model.mse(input))
}
