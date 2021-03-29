package it.unibo.learning

import smile.data.DataFrame
import smile.regression.DataFrameRegression
import smile.write

import scala.util.Try

object RegressionImplicits {
  implicit class RichRegression(r: DataFrameRegression) {
    def store(path: String): Try[Unit] = Try {
      write.xstream(r, path)
    }

    def mse(dataFrame: DataFrame): Double = {
      val expected = r.formula().y(dataFrame)
      val result = r.predict(dataFrame)
      val reference = expected.toDoubleArray
      result
        .zip(reference)
        .map { case (a, b) => Math.pow(a - b, 2) }
        .sum / result.size
    }
  }
}
