const expect = require('chai').expect;

const Layer = require('../layer');

describe('Layer', () => {

    describe('constructor', () => {

        it('creates (width) neurons with weight width of (inputWidth)', () => {

            const l = Layer(3, 2);

            expect(l.neurons.length).to.equal(2);

            expect(l.neurons[0].weight.length).to.equal(3);
            expect(l.neurons[1].weight.length).to.equal(3);
        });
    });

    describe('get params', () => {

        it('retrieves an array of the combined weight and bias of all neurons', () => {

            const l = Layer(3, 2);

            const params = l.params;

            expect(params.length).to.equal(8);
            expect(l.neurons[0].bias).to.equal(params[0]);
            expect(l.neurons[0].weight[0]).to.equal(params[1]);
            expect(l.neurons[0].weight[1]).to.equal(params[2]);
            expect(l.neurons[0].weight[2]).to.equal(params[3]);
            expect(l.neurons[1].bias).to.equal(params[4]);
            expect(l.neurons[1].weight[0]).to.equal(params[5]);
            expect(l.neurons[1].weight[1]).to.equal(params[6]);
            expect(l.neurons[1].weight[2]).to.equal(params[7]);
        });
    });

    describe('set params', () => {

        it('sets from array the weight and bias of all neurons', () => {

            const l = Layer(3, 2);

            l.params = [0, 1, 2, 3, 4, 5, 6, 7];

            expect(l.neurons[0].bias).to.equal(0);
            expect(l.neurons[0].weight[0]).to.equal(1);
            expect(l.neurons[0].weight[1]).to.equal(2);
            expect(l.neurons[0].weight[2]).to.equal(3);
            expect(l.neurons[1].bias).to.equal(4);
            expect(l.neurons[1].weight[0]).to.equal(5);
            expect(l.neurons[1].weight[1]).to.equal(6);
            expect(l.neurons[1].weight[2]).to.equal(7);
        });
    });

    describe('calc', () => {

        const layer = Layer(3, 2);

        it('creates array of calc outputs from all neurons', () => {

            layer.params = [ 0, 1, 1, 1, 0, 1, 1, 1];

            const result = layer.calc([2, 2, 2]);

            // weighted Sum = 2*1 + 2*1 + 2*1 = 6
            // + bias = 6
            // sigmoid(6) = 0.9975273768433653
            expect(result).to.deep.equal([ 0.9975273768433653, 0.9975273768433653 ]);
        });

        it('stores the output from the last calc', () => {
            expect(layer.output).to.deep.equal([ 0.9975273768433653, 0.9975273768433653 ]);
        });
    });

});
