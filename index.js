const _ = require('lodash');
const mnist = require('mnist');

const Network = require('./network');
const util = require('./util');

const trainingSetSize = 800;
const testSetSize = 200;

const learningRate = 0.5;
const targetCost = 0.015;
const maxEpochs = 1000;

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

// const updateProgress = util.progressTracker();

while( (epoch++ < maxEpochs) && (currentCost > targetCost)) {

    _.each(trainingSet, (lesson, index) => {
        util.backPropagateTuned(nw, lesson, { learningRate });

        // (index % 10 === 0) && updateProgress({
        //     epoch,
        //     epochProgress: index,
        //     lastCost: util.evaluateCost(nw, [lesson])
        // });
    });

    currentCost = util.evaluateCost(nw, _.sampleSize(testSet, 50));
    process.stdout.write(`Epoch: ${ epoch }, current cost: ${ currentCost }   ` + '\r');

    stats.log({ epoch, currentCost });
}

// evaluate final accuracy
const accuracy = util.decimalToPercent(util.calcAccuracy(nw, testSet));
console.log(`\nfinal Accuracy: ${ accuracy }`);

stats.dump({ accuracy });
