const tf = require('@tensorflow/tfjs')
const _ = require('lodash')

class LinearRegression {
    constructor(features, labels, options) {
        this.features = this.processFeature(features)
        this.labels = tf.tensor(labels)

        // assigns a default value to options so that when options is not input, the whole thing 
        // doesn't return undefined
        this.options = Object.assign(
            { learningRate: 0.1, iterations: 1000 }, 
            options
        )
        // the number of columns indicate the number of features, therefore this.feature.shape[1] will equal to the number of weights
        this.weights = tf.zeros([this.features.shape[1], 1])
    }

    gradientDescent() {
        const currentGuess = this.features.matMul(this.weights)
        const differences = currentGuess.sub(this.labels)
        const slope = this.features
            .transpose()
            .matMul(differences)
            .div(this.features.shape[0])
        
        this.weights = this.weights.sub(slope.mul(this.options.learningRate))
    }   

    train() {
        for (let i = 0; i < this.options.iterations; i++) {
            this.gradientDescent()
        }
    }

    test(testFeatures, testLabels) {
        testFeatures = this.processFeature(testFeatures)
        testLabels = tf.tensor(testLabels)

        const predictions = testFeatures.matMul(this.weights)
        const res = testLabels
            .sub(predictions)
            .pow(2)
            .sum()
            .arraySync()
        
        const tot = testLabels
            .sub(testLabels.mean())
            .pow(2)
            .sum()
            .arraySync()

        return 1 - res / tot
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
}

// class LinearRegression {
//     constructor(features, labels, options) {
//         this.features = features
//         this.labels = labels
//         // assigns a default value to options so that when options is not input, the whole thing 
//         // doesn't return undefined
//         this.options = Object.assign(
//             { learningRate: 0.1, iterations: 1000 }, 
//             options
//         )

//         this.m = 0;
//         this.b = 0;
//     }

//     gradientDescent() {
//         const currentGuessForMPG = this.features.map(row => {
//             return this.m * row[0] + this.b
//         })

//         const bSlope = _.sum(currentGuessForMPG.map((guess, i ) => {
//             return guess - this.labels[i][0]
//         })) * 2 / this.features.length

//         const mSlope = _.sum(currentGuessForMPG.map((guess, i) => {
//             return -1 * this.features[i][0] * (this.labels[i][0] - guess)
//         })) * 2 / this.features.length

//         this.m = this.m - mSlope * this.options.learningRate
//         this.b = this.b - bSlope * this.options.learningRate 
//     }

//     train() {
//         for (let i = 0; i < this.options.iterations; i++) {
//             this.gradientDescent()
//         }
//     }
// }

module.exports = LinearRegression