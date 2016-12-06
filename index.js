const _ = require('lodash');
const mnist = require('mnist');

const Network = require('./network');
const util = require('./util');

const trainingSetSize = 800;
const testSetSize = 200;

const learningRate = 0.5;
const targetCost = 0.015;
const maxEpochs = 1000;
const batchSize = 10;

// const trainingSet = [
//     { input: [ 0, 0 ], output: [ 0, 1 ] },
//     { input: [ 0, 1 ], output: [ 1, 0 ] },
//     { input: [ 1, 0 ], output: [ 1, 0 ] },
//     { input: [ 1, 1 ], output: [ 0, 1 ] }
// ];

const stats = util.statTracker();

process.on('exit', stats.dump);

const {
    training: trainingSet,
    test: testSet
} = mnist.set(trainingSetSize, testSetSize);

const nw = Network(
    trainingSet[0].input.length, // input Width
    [
        100, // hidden layer width
        trainingSet[0].output.length // output width
    ]
);

console.log(`initial Accuracy: ${ util.decimalToPercent(util.calcAccuracy(nw, testSet)) }`);

let currentCost = 10e10;
let epoch = 0;
let batch;

while( (epoch++ < maxEpochs) && (currentCost > targetCost)) {

    batch = _.sampleSize(trainingSet, batchSize);

    // TODO: this is basically online learning, i'm not properly batching.
    _.each(batch, lesson => {
        util.backPropagate(nw, lesson.input, lesson.output, learningRate);
    });

    currentCost = util.evaluateCost(nw, batch);

    process.stdout.write(`Epoch: ${ epoch }, current cost: ${ currentCost }   ` + '\r');
    stats.log({ epoch, currentCost });
}

// evaluate final accuracy
const accuracy = util.decimalToPercent(util.calcAccuracy(nw, testSet));
console.log(`\nfinal Accuracy: ${ accuracy }`);

stats.dump({ accuracy });
