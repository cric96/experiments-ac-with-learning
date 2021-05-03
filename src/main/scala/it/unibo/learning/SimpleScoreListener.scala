package it.unibo.learning

import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.optimize.api.BaseTrainingListener

class SimpleScoreListener extends BaseTrainingListener {
  private var epoch   = 0
  private var score   = 0.0
  private var howMany = 0

  override def iterationDone(model: Model, iteration: Int, epoch: Int): Unit = {
    this.score += model.score
    howMany    += 1
    if (this.epoch != epoch) {
      this.epoch = epoch
      println(s"Score at iteration $iteration is ${this.score / howMany};; epoch $epoch")
      this.howMany = 0
      this.score = 0
    }
  }
}
