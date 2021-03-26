package it.unibo.visualization
import smile.data.`type`.{DataTypes, StructField, StructType}
import smile.plot.Render.renderCanvas
import smile.plot._
import smile.plot.swing._
import smile.read

object PlotData extends App {
  val input = read.csv(
    file = "output.csv",
    schema = new StructType(
      new StructField("y", DataTypes.DoubleType),
      new StructField("target", DataTypes.DoubleType),
      new StructField("min", DataTypes.DoubleType),
    )
  )
  val x = input.select(0, 1, 2).toArray

  val result = surface(x)
  show(result)
  println(input)
}
