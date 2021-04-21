package it.unibo.learning

import smile.data.DataFrame
import smile.data.Tuple
import smile.data.`type`.DataTypes
import smile.data.`type`.StructField
import smile.data.`type`.StructType
import smile.data.formula.Formula
import smile.data.formula._
import smile.read

import scala.language.postfixOps

object Dataset {

  implicit class RichTuple(tuple: Tuple) {
    def min: Double = tuple.getDouble("min")
  }

  val schema: StructType = new StructType(
    new StructField("y", DataTypes.DoubleType),
    new StructField("target", DataTypes.DoubleType),
    new StructField("min", DataTypes.DoubleType)
  )

  def createTuple(target: Double, minHop: Double): Tuple = {
    Tuple.of(
      Array[Double](target, minHop),
      new StructType(
        new StructField("target", DataTypes.DoubleType),
        new StructField("min", DataTypes.DoubleType)
      )
    )
  }

  def load(file: String): DataFrame =
    read.csv(file = file, schema = Dataset.schema)

  val formula: Formula = "y" ~

}
