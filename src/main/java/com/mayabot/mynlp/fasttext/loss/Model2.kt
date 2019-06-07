package com.mayabot.mynlp.fasttext.loss

import com.carrotsearch.hppc.IntArrayList
import com.mayabot.blas.FloatMatrix
import com.mayabot.blas.vector.FloatArrayVector
import com.mayabot.mynlp.fasttext.FloatIntPair
import kotlin.random.Random

inline fun IntArrayList.forEach2(action: (num: Int) -> Unit) {
    val buffer = this.buffer
    var i = 0
    val size = this.size()
    while (i < size) {
        action(buffer[i])
        i++
    }
}

typealias Predictions = MutableList<FloatIntPair>


class Model2(
        val wi: FloatMatrix,
        val wo: FloatMatrix,
        val loss: Loss,
        val normalizeGradient: Boolean
) {

    companion object {
        val kUnlimitedPredictions: Int = -1
        val kAllLabelsAsTarget = -1
    }

    fun computeHidden(input: IntArrayList, state: State) {
        val hidden = state.hidden
//        checkArgument(hidden.length() == hsz)
        hidden.zero()

        input.forEach2 { row ->
            hidden += wi[row]
        }

        //长度归一化
        hidden *= (1.0f / input.size())
    }

    /**
     * 预测分类结果
     *
     * 预测过程。。。
     *
     * @param input 输入的词的下标
     *
     */
    fun predict(input: IntArrayList,
                k: Int,
                threshold: Float,
                heap: Predictions,
                state: State
    ) {
        val kk = if (k == kUnlimitedPredictions) {
            // output size
            wo.rows()
        } else{
            k
        }
        if (kk == 0) {
            throw RuntimeException("k needs to be 1 or higher")
        }

        computeHidden(input,state)

        loss.predict(k,threshold,heap,state)
    }

    fun update(input: IntArrayList,
               targets:IntArrayList,
               targetIndex:Int,
               lr:Float,
               state:State){
        if (input.size() == 0) {
            return
        }

        computeHidden(input,state)
        val grad = state.grad
        grad.zero()

        val lossValue = loss.forward(targets,targetIndex,state,lr,true)

        state.incrementNExamples(lossValue)

        if (normalizeGradient) {
            grad *= (1.0f/input.size())
        }

        input.forEach2 { i->
            wi.addVectorToRow(grad,i,1.0f)
        }
    }

    class State(hiddenSize: Int, outputSize: Int, seed: Int) {
        private var lossValue = 0.0f
        private var nexamples = 0

        val hidden = FloatArrayVector(hiddenSize)
        val output = FloatArrayVector(outputSize)
        val grad = FloatArrayVector(hiddenSize)
        val rng = Random(seed)

        fun getLoss() = lossValue / nexamples

        fun incrementNExamples(loss: Float) {
            lossValue += loss
            nexamples++
        }
    }

}