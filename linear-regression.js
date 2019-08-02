const tf = require('@tensorflow/tfjs')
const _ = require('lodash')

class LinearRegression {
    constructor(features, labels, options) {
        this.features = tf.tensor(features)
        this.labels = tf.tensor(labels)

        this.features = tf
            .ones([this.features.shape[0], 1])
            .concat(this.features, 1)

        // assigns a default value to options so that when options is not input, the whole thing 
        // doesn't return undefined
        this.options = Object.assign(
            { learningRate: 0.1, iterations: 1000 }, 
            options
        )
        
        this.weights = tf.zeros([2, 1])

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