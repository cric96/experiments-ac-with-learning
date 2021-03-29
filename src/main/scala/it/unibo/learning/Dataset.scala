package it.unibo.learning

import smile.data.{DataFrame, Tuple}
import smile.data.`type`.{DataTypes, StructField, StructType}
import smile.data.formula.Formula
import smile.read
import smile.data.formula._
import scala.language.postfixOps

object Dataset {
  val schema: StructType = new StructType(
    new StructField("y", DataTypes.DoubleType),
    new StructField("target", DataTypes.DoubleType),
    new StructField("min", DataTypes.DoubleType),
  )

  def createTuple(target: Double, minHop: Double): Tuple = {
    Tuple.of(
      Array[Double](target, minHop),
      new StructType(
        new StructField("target", DataTypes.DoubleType),
        new StructField("min", DataTypes.DoubleType),
      )
    )
  }

  def load(file: String): DataFrame = {
    read.csv(file = file, schema = Dataset.schema)
  }

  val formula: Formula = "y" ~

}
