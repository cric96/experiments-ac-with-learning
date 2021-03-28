package it.unibo.learning

import smile.data.`type`.{DataTypes, StructField, StructType}

object Dataset {
  val schema = new StructType(
    new StructField("y", DataTypes.DoubleType),
    new StructField("target", DataTypes.DoubleType),
    new StructField("min", DataTypes.DoubleType),
  )
}
