package it.unibo.learning

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import smile.data.DataFrame
import smile.data.Tuple
import smile.data.`type`.DataTypes
import smile.data.`type`.StructField
import smile.data.`type`.StructType
import smile.data.formula.Formula
import smile.data.formula._
import smile.read

object Dataset {

  implicit class RichTuple(tuple: Tuple) {
    def min: Double         = tuple.getDouble("min")
    def asNDArray: INDArray = new NDArray(Array(tuple.getDouble("target").toFloat, tuple.min.toFloat))
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

  val formula: Formula = "y".~()

}
