const expect = require('chai').expect;
const _ = require('lodash');

const util = require('../util');
const Network = require('../network');

const TOLERANCE = 0.0001;

describe('util', () => {

    describe('vectorDistance', () => {

        it('throws if the vectors are of unequal length', () => {
           expect(() => util.vectorDistance([1], [1, 2])).to.throw(Error);
        });

        it('calculates the Euclidian distance between two vectors', () => {

            const a = [0, 0];
            const b = [1, 1];

            const result = (util.vectorDistance(a, b));

            expect(result).to.equal(Math.sqrt(2));
        });
    });

    describe('avgVectorDistance', () => {

        it('calculates the average Euclidian distance between two sets of vectors', () => {

            const vectorPairs = [
                [[0, 0], [0, 0]],
                [[1, 1], [3, 3]]
            ];

            const result = (util.avgVectorDistance(vectorPairs));

            expect(result).to.equal(
                (Math.sqrt(0 + 0) + Math.sqrt(4 + 4)) / 2
            );
        });
    });

    describe('evaluateCost', () => {

        it(`runs all training sets through a network and calcs the avg dist between 
            the result and the training sets expected result`, () => {

            const net = {
                calc: x => [x[0] * x[1] * 2,  x[0] * x[1] * 3]
            };

            const trainingSet = [
                { input: [0, 0], output: [0, 0] },
                { input: [1, 1], output: [2, 3] },
                { input: [2, 2], output: [8, 13] }
            ];

            const result = (util.evaluateCost(net, trainingSet));

            expect(result).to.equal(
                1 / 3
            );
        });
    });

    describe('computeNumericalGradient', () => {
        it('calculates the gradient of a function at a specific input value by numerical methods', () => {

            const func = input => input.map((i, idx) => idx * i);
            const position = [1, 1, 1, 1];

            const result = util.computeNumericalGradient(func, position);

            expect(result[0]).to.be.approximately(0, TOLERANCE);
            expect(result[1]).to.be.approximately(1, TOLERANCE);
            expect(result[2]).to.be.approximately(2, TOLERANCE);
            expect(result[3]).to.be.approximately(3, TOLERANCE);
        });
    });

    describe('backPropogate', () => {
        it('updates the weights of a network given the error', () => {

            const network = new Network(2, [2, 2]);

            network.params = [
                0.35, 0.15, 0.20,
                0.35, 0.25, 0.30,
                0.60, 0.40, 0.45,
                0.60, 0.50, 0.55
            ];

            const input = [ 0.05, 0.10 ];
            const expected = [ 0.01, 0.99 ];

            const learning_rate = 0.5;

            util.backPropagate(network, input, expected, learning_rate);

            const expectedParams = [
                0.3456, 0.14978, 0.1995,
                0.3450, 0.24975, 0.2995,
                0.53075, 0.35891, 0.4086,
                0.61905, 0.51130, 0.5613
            ];

            _.each(network.params, (param, i) => {
                expect(param).to.be.approximately(expectedParams[i], TOLERANCE);
            });
        });
    });

    describe('indexOfMax', () => {
        it('returns the index of the entry in an array with the highest value', () => {

            const test = [ 0, 0, 99, 100, 98, 0 ];
            const expected = 3;

            expect(util.indexOfMax(test)).to.equal(expected);

        });
    });

    describe('decimalToPercent', () => {
        it('accurately converts a decimal fraction to a human readable percent of the desired precision', () => {

            const test = 0.987654;
            const expected = '98.765%';

            expect(util.decimalToPercent(test, 3)).to.equal(expected);

        });
    });

    describe('calcAccuracy', () => {
        it('evaluates the fraction of correct classifications of a network with a specified test set', () => {

            const testSet = [
                { input: [ 0 ], output: [ 1, 0, 0 ] },
                { input: [ 1 ], output: [ 0, 1, 0 ] },
                { input: [ 2 ], output: [ 1, 0, 0 ] }
            ];

            const testNet = { calc: input => [0, 0, 0].map((val, idx) => input === idx) };

            const expected = 0.6666666666666666;

            expect(util.calcAccuracy(testNet, testSet)).to.equal(expected);

        });
    });
});
