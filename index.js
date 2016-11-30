const _ = require('lodash');

const Network = require('./network');
const util = require('./util');

const nw = Network(2, [2, 2, 2]);

const trainingSet1 = [
    { input: [ 0, 0 ], output: [ 0, 1 ] },
    { input: [ 0, 1 ], output: [ 1, 0 ] },
    { input: [ 1, 0 ], output: [ 1, 0 ] },
    { input: [ 1, 1 ], output: [ 0, 1 ] }
];

console.log('initial cost: ' + util.evaluateCost(nw, trainingSet1));

const learningRate = 0.5;

_.times(20000, () => {

    _.each(trainingSet1, lesson => {
            util.backPropagate(nw, lesson.input, lesson.output, learningRate);
        }
    );

    process.stdout.write('current cost: ' + util.evaluateCost(nw, trainingSet1) + '\r');
});
console.log('');

_.each(trainingSet1, lesson => {
        console.log('input of ' + lesson.input
            + ' gives output of ' + nw.calc(lesson.input)
            + ', expected output ' + lesson.output
        );
    }
);
