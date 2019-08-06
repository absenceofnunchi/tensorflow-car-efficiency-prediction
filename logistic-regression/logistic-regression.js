const tf = require('@tensorflow/tfjs')
const _ = require('lodash')

class LogististicRegression {
    constructor(features, labels, options) {
        this.features = this.processFeature(features)
        this.labels = tf.tensor(labels)
        // cross entropy history
        this.costHistory = []

        // assigns a default value to options so that when options is not input, the whole thing 
        // doesn't return undefined
        this.options = Object.assign(
            { learningRate: 0.1, iterations: 1000, decisionBoundary: 0.5 }, 
            options
        )
        // the number of columns indicate the number of features, therefore this.feature.shape[1] will equal to the number of weights
        this.weights = tf.zeros([this.features.shape[1], 1])
    }

    gradientDescent(features, labels) {
        const currentGuess = features.matMul(this.weights).sigmoid()
        const differences = currentGuess.sub(labels)
        const slope = features
            .transpose()
            .matMul(differences)
            .div(features.shape[0])
        
        this.weights = this.weights.sub(slope.mul(this.options.learningRate))
    }   

    train() {
        const batchQuantity = Math.floor(
            this.features.shape[0] / this.options.batchSize
        )

        for (let i = 0; i < this.options.iterations; i++) {
            for (let j = 0; j < batchQuantity; j++) {
                const startIndex = j * this.options.batchSize
                const { batchSize } = this.options
                const featureSlice = this.features.slice(
                    [startIndex, 0],
                    [batchSize, -1]
                )
                const labelSlice = this.labels.slice(
                    [startIndex, 0], 
                    [batchSize, -1]
                )

                this.gradientDescent(featureSlice, labelSlice)
            }
            console.log(`Learning rate: ${this.options.learningRate}`)
            this.recordCost()
            this.updateLearningRate()
        }
    }

    predict(observations) {
        return this.processFeature(observations)
            .matMul(this.weights)
            .sigmoid()
            .greater(this.options.decisionBoundary) // converts the tensors to boolean
            .cast('float32') // converts the tensors back to numbers
    }

    test(testFeatures, testLabels) {
        const predictions = this.predict(testFeatures).round()
        testLabels = tf.tensor(testLabels)

        const incorrect = predictions
            .sub(testLabels)
            .abs()
            .sum()
            .arraySync()

        return (predictions.shape[0] - incorrect) / predictions.shape[0]
    }

    processFeature(features) {
        features = tf.tensor(features)

        if(this.mean && this.variance) {
            features = features.sub(this.mean).div(this.variance.pow(0.5))
        } else {
            features = this.standardize(features)
        }
        // ones: the first parameter indicates the row, the second indicates column
        // shape: 0 indicates the number of rows in this.features
        features = tf.ones([features.shape[0], 1]).concat(features, 1)

        return features
    }

    standardize(features) {
        const { mean, variance } = tf.moments(features, 0)
        const m = mean.arraySync()
        
        this.mean = mean
        this.variance = variance

        return features.sub(mean).div(variance.pow(0.5))
    }

    recordCost() {
        const guesses = this.features.matMul(this.weights).sigmoid()
        const termOne = this.labels.transpose().matMul(guesses.log())
        const termTwo = this.labels
            .mul(-1)
            .add(1)
            .transpose()
            .matMul(
                guesses
                    .mul(-1)
                    .add(1)
                    .log()
            )
        
        const cost = termOne.add(termTwo)
            .div(this.features.shape[0])
            .mul(-1)
            .arraySync()[0][0]
        
        this.costHistory.unshift(cost)
    }

    updateLearningRate() {
        if (this.costHistory.length < 2) {
            return
        }

        if(this.costHistory[0] > this.costHistory[1]) {
            this.options.learningRate /= 2
        } else {
            this.options.learningRate *= 1.05
        }
    }
}

module.exports = LogististicRegression
