package com.mayabot.mynlp.fasttext.loss

import com.carrotsearch.hppc.IntArrayList
import com.mayabot.mynlp.fasttext.BaseModel

const val SIGMOID_TABLE_SIZE = 512
const val MAX_SIGMOID = 8
const val LOG_TABLE_SIZE = 512
const val NEGATIVE_TABLE_SIZE = 10000000


abstract class Loss{
    abstract fun predict(k: Int, threshold: Float, heap: Predictions, state: Model2.State)
    abstract fun forward(targets: IntArrayList, targetIndex: Int, state: Model2.State, lr: Float, b: Boolean): Float

    companion object{
         private val tSigmoid: FloatArray = FloatArray(SIGMOID_TABLE_SIZE + 1) { i ->
            val x = (i * 2 * MAX_SIGMOID).toFloat() / SIGMOID_TABLE_SIZE - MAX_SIGMOID
            (1.0f / (1.0f + Math.exp((-x).toDouble()))).toFloat()
        }

         private val tLog: FloatArray = FloatArray(LOG_TABLE_SIZE + 1) { i ->
            val x = (i.toFloat() + 1e-5f) / LOG_TABLE_SIZE
            Math.log(x.toDouble()).toFloat()
        }

        fun log(x: Float): Float {
            if (x > 1.0f) {
                return 0.0f
            }
            val i = (x * LOG_TABLE_SIZE).toInt()
            return tLog[i]
        }

        fun sigmoid(x: Float): Float {
            return when {
                x < -MAX_SIGMOID -> 0.0f
                x > MAX_SIGMOID -> 1.0f
                else -> {
                    val i = ((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID.toFloat() / 2f).toInt()
                    tSigmoid[i]
                }
            }
        }
    }

}
//
//abstract class BinaryLogisticLoss:Loss(){
//    fun binaryLogistic(
//            target: Int,
//            label: Boolean,
//            lr: Float): Float {
//        val score = BaseModel.sigmoid(outputMatrix[target] * hidden)
//        val alpha = lr * ((if (label) 1.0f else 0.0f) - score)
//        grad += alpha to outputMatrix[target]
//
//        outputMatrix[target] += alpha to hidden
//
//        return if (label) {
//            -BaseModel.log(score)
//        } else {
//            -BaseModel.log(1.0f - score)
//        }
//    }
//}
