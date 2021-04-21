package it.unibo.learning

import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.optimize.api.BaseTrainingListener

class SimpleScoreLister extends BaseTrainingListener {

  override def iterationDone(model: Model, iteration: Int, epoch: Int): Unit = {
    val score = model.score
    println(s"Score at iteration $iteration is $score")
  }
}
