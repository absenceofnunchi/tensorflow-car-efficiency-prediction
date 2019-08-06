require('@tensorflow/tfjs-node')
const tf =require('@tensorflow/tfjs')
const loadCSV = require('../load-csv')
const LogicalRegression = require('./logistic-regression')
const plot = require('node-remote-plot')
const _ = require('lodash')

const { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv',{
    dataColumns: [
        'horsepower',
        'displacement',
        'weight'
    ],
    labelColumns: ['mpg'],
    shuffle: true,
    splitTest: 50,
    converters: {
        mpg: value => {
            const mpg = parseFloat(value)
            if (mpg < 15) {
                return [1, 0, 0]
            } else if (mpg < 30) {
                return [0, 1, 0]
            } else {
                return [0, 0, 1]
            }
        }
    }
})

const regression = new LogicalRegression(features, _.flatMap(labels), {
    learningRate: 0.5,
    iterations: 100,
    batchSize: 40,
    decisionBoundary: 0.6
})

regression.train()
regression.predict([[215, 440, 2.16]]).print()
plot({
    x: regression.costHistory.reverse()
})