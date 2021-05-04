package it.unibo.casestudy

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist.ScafiAlchemistSupport
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.cpu.nativecpu.NDArray

class NetworkEvaluator extends AggregateProgram with StandardSensors with ScafiAlchemistSupport with FieldUtils {
  private val maxValue    = 1000.0
  private val cnnNetwork  = ModelSerializer.restoreMultiLayerNetwork("src/main/resources/network")
  private val lstmNetwork = ModelSerializer.restoreMultiLayerNetwork("src/main/resources/lstmnetwork")

  override def main(): Any = {
    val reference = hopCountLike(_.min + 1)
    val networkResult = hopCountLike { elements =>
      val ndarray = new NDArray(elements.map(_.toFloat).toArray, Array(1, 1, elements.size))
      cnnNetwork.output(ndarray).getDouble(0L)
    }
    val lstmResult = hopCountLike { elements =>
      val ndarray = new NDArray(elements.map(_.toFloat).toArray, Array(1, 1, elements.size))
      val result  = lstmNetwork.output(ndarray)
      result.getDouble((elements.size - 1).toLong)
    }
    val cnnSquaredError  = Math.pow(networkResult - reference, 2)
    val lstmSquaredError = Math.pow(lstmResult - reference, 2)
    node.put("reference", reference)
    node.put("cnn_network", cnnSquaredError)
    node.put("lstm_network", lstmSquaredError)
  }

  private def hopCountLike(expr: Iterable[Double] => Double): Double =
    rep(maxValue) { data =>
      val values = excludingSelf.reifyField(nbr(data)).values
      val result = if (values.nonEmpty) expr(values).toInt else data.toInt
      mux(isTarget)(0)(result) //not right
    //right way : result
    }

  private def isTarget: Boolean = target == 1.0

  private def target: Double = sense[Double]("target")
}
