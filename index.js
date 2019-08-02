require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const loadCSV = require('./load-csv')
const LinearRegression = require('./linear-regression')

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower'],
    labelColumns: ['mpg']
})

const regression = new LinearRegression(features, labels, {
    learningRate: 0.001,
    iterations: 1
})

regression.train()
regression.weights.array().then((results) => {
    console.log(`Updated M is ${results[1][0]}. Updated B is ${results[0][0]}`)
})
