const expect = require('chai').expect;

const Network = require('../network');

describe('Network', () => {

    describe('constructor', () => {

        it('creates layers with the first having inputwidth of inputwidth and the rest'
            + 'having inputwidth of prev layers width', () => {

            const nw = Network(3, [3, 2]);

            expect(nw.layers.length).to.equal(2);

            expect(nw.layers[0].neurons.length).to.equal(3);
            expect(nw.layers[1].neurons.length).to.equal(2);

            expect(nw.layers[0].neurons[0].weight.length).to.equal(3);
            expect(nw.layers[1].neurons[0].weight.length).to.equal(3);
        });
    });

    describe('get params', () => {

        it('retrieves an array of the combined weight and bias of all neurons', () => {

            const nw = Network(2, [2, 1]);

            const params = nw.params;

            expect(params.length).to.equal(9);
            expect(nw.layers[0].neurons[0].bias).to.equal(params[0]);
            expect(nw.layers[0].neurons[0].weight[0]).to.equal(params[1]);
            expect(nw.layers[0].neurons[0].weight[1]).to.equal(params[2]);
            expect(nw.layers[0].neurons[1].bias).to.equal(params[3]);
            expect(nw.layers[0].neurons[1].weight[0]).to.equal(params[4]);
            expect(nw.layers[0].neurons[1].weight[1]).to.equal(params[5]);
            expect(nw.layers[1].neurons[0].bias).to.equal(params[6]);
            expect(nw.layers[1].neurons[0].weight[0]).to.equal(params[7]);
            expect(nw.layers[1].neurons[0].weight[1]).to.equal(params[8]);
        });
    });

    describe('set params', () => {

        it('sets from array the weight and bias of all neurons', () => {

            const nw = Network(2, [2, 1]);

            nw.params = [0, 1, 2, 3, 4, 5, 6, 7, 8];

            expect(nw.layers[0].neurons[0].bias).to.equal(0);
            expect(nw.layers[0].neurons[0].weight[0]).to.equal(1);
            expect(nw.layers[0].neurons[0].weight[1]).to.equal(2);
            expect(nw.layers[0].neurons[1].bias).to.equal(3);
            expect(nw.layers[0].neurons[1].weight[0]).to.equal(4);
            expect(nw.layers[0].neurons[1].weight[1]).to.equal(5);
            expect(nw.layers[1].neurons[0].bias).to.equal(6);
            expect(nw.layers[1].neurons[0].weight[0]).to.equal(7);
            expect(nw.layers[1].neurons[0].weight[1]).to.equal(8);
        });
    });

    describe('calc', () => {

        it('creates array of calc outputs from all neurons', () => {

            const nw = Network(3, [2]);
            nw.params = [ 0, 1, 1, 1, 0, 1, 1, 1];

            const result = nw.calc([2, 2, 2]);

            // weighted Sum = 2*1 + 2*1 + 2*1 = 6
            // + bias = 6
            // sigmoid(6) = 0.9975273768433653
            expect(result).to.deep.equal([ 0.9975273768433653, 0.9975273768433653 ]);
        });
    });

});
