package it.unibo.learning
import smile.data.Tuple
import smile.data.`type`.{DataTypes, StructField, StructType}
import smile.data.formula._
import smile.read
import smile.regression._

import scala.language.postfixOps

object LinearModel extends App {
  val input = read.csv(
    file = "output.csv",
    schema = new StructType(
      new StructField("y", DataTypes.DoubleType),
      new StructField("target", DataTypes.DoubleType),
      new StructField("min", DataTypes.DoubleType),
    )
  )
  val formula: Formula = "y" ~
  val model = ridge(formula, input, 0.0001)

  def createTuple(a: (Double, Double)): Tuple = Tuple.of(
    Array[Double](a._1, a._2),
    new StructType(
      new StructField("target", DataTypes.DoubleType),
      new StructField("min", DataTypes.DoubleType),
    )
  )
  def createTest(in: Double): Tuple = createTuple(0, in)

  def createSource(in: Double): Tuple = createTuple(1.0, in)

  //it learns min + 1:
  println(model.predict(createTest(1.0)))
  println(model.predict(createTest(100.0)))
  //the error with high number is consistent..
  println(model.predict(createTest(1000.0)))
  //it doesn't learn target => 0.0 (it learns min - 1)
  println(model.predict(createSource(0.0)))
  println(model.predict(createSource(100.0)))
}
