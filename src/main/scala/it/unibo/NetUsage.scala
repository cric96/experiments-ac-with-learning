package it.unibo

import org.nd4j.linalg.cpu.nativecpu.NDArray

object NetUsage extends App {
  val net = DeepNet(inputSize = 2, outputSize = 1, 5 :: Nil)
  net.init()
  new NDArray(Array(10.0f, 5.0f))
  println(net.predict(new NDArray(Array(10.0f, 5.0f))).mkString)
}
