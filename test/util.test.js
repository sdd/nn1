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

            const A = [[0, 0], [0, 0]];
            const B = [[1, 1], [2, 2]];

            const result = (util.avgVectorDistance(A, B));

            expect(result).to.equal(
                (Math.sqrt(1 + 1) + Math.sqrt(4 + 4)) / 2
            );
        });
    });

    describe.only('evaluateCost', () => {

        it(`runs all training sets through a network and calcs the avg dist between 
            the result and the training sets expected result`, () => {

            const net = {
                calc: x => [x[0] * x[1] * 2,  x[0] * x[1] * 3]
            };

            const trainingSet = [
                { input: [0, 0], output: [0, 0] },
                { input: [1, 1], output: [2, 3] },
                { input: [2, 2], output: [8, 12] }
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
});
