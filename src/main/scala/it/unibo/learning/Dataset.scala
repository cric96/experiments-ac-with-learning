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

  implicit class RichTuple(t: Tuple) {
    def min: Double         = t.getDouble("min")
    def max: Double         = t.getDouble("max")
    def avg: Double         = t.getDouble("avg")
    def isTarget: Double    = t.getDouble("isTarget")
    def isNotTarget: Double = t.getDouble("isNotTarget")
    def y: Double           = t.getDouble("y")
    def asNDArray: INDArray = new NDArray(Array(t.min, t.max, t.avg, t.isTarget, t.isNotTarget).map(_.toFloat))
  }

  val schema: StructType = new StructType(
    new StructField("min", DataTypes.DoubleType),
    new StructField("max", DataTypes.DoubleType),
    new StructField("avg", DataTypes.DoubleType),
    new StructField("isTarget", DataTypes.DoubleType),
    new StructField("isNotTarget", DataTypes.DoubleType),
    new StructField("y", DataTypes.DoubleType)
  )

  val inputSchema: StructType = new StructType(schema.fields.reverse.tail.reverse: _*)

  def createTuple(
      min: Double,
      max: Double,
      avg: Double,
      isTarget: Double,
      isNotTarget: Double
  ): Tuple = {
    Tuple.of(
      Array[Double](min, max, avg, isTarget, isNotTarget),
      inputSchema
    )
  }

  def load(file: String): DataFrame =
    read.csv(file = file, schema = Dataset.schema)

  val formula: Formula = "y".~()

}
