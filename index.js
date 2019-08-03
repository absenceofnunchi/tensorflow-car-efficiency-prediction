require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const loadCSV = require('./load-csv')
const LinearRegression = require('./linear-regression')

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'displacement', 'weight', 'acceleration', 'cylinders', 'modelyear'],
    labelColumns: ['mpg']
})

const regression = new LinearRegression(features, labels, {
    learningRate: 0.1,
    iterations: 1000
})

regression.train()
regression.weights.array().then((results) => {
    console.log(`Updated M is ${results[1][0]}. Updated B is ${results[0][0]}`)
})

const r2 = regression.test(testFeatures, testLabels)
console.log(`r2 is ${r2}`)
