const _ = require('lodash');

const Network = require('./network');
const util = require('./util');

const nw = Network(2, [2, 2, 2]);

const trainingSet1 = [
    { input: [ 0, 0 ], expected: [ 0, 1 ] },
    { input: [ 0, 1 ], expected: [ 1, 0 ] },
    { input: [ 1, 0 ], expected: [ 1, 0 ] },
    { input: [ 1, 1 ], expected: [ 0, 1 ] }
];

const p = nw.params;
console.log('p: ' + p);
nw.params = p.map(i => i + 1);
const p2 = nw.params;
console.log('p2: ' + p2);

console.log('cost: ' + util.evaluateCost(nw, trainingSet1));

_.times(100, () => {

    _.each(trainingSet1, trainingExample => {
            util.backPropagate(nw, trainingExample.input, trainingExample.expected, learningRate);
        }
    );

    console.log('cost: ' + util.evaluateCost(nw, trainingSet1));
});

