const expect = require('chai').expect;

const Neuron = require('../neuron');

describe('Neuron', () => {

    describe('constructor', () => {

        it('handles being passed just an input width value', () => {

            const result = Neuron(5);

            expect(result.weight.length).to.equal(5);
            expect(result.weight[0]).to.be.within(-10, 10);
            expect(result.weight[1]).to.be.within(-10, 10);
            expect(result.weight[2]).to.be.within(-10, 10);
            expect(result.weight[3]).to.be.within(-10, 10);
            expect(result.weight[4]).to.be.within(-10, 10);

            expect(result.bias).to.be.within(-10, 10);
        });
    });

    describe('get params', () => {

        it('retrieves an array of weight and bias', () => {

            const n = Neuron(5);

            const [ bias, ...weight ] = n.params;

            expect(weight.length).to.equal(5);
            expect(weight[0]).to.be.within(-10, 10);
            expect(weight[1]).to.be.within(-10, 10);
            expect(weight[2]).to.be.within(-10, 10);
            expect(weight[3]).to.be.within(-10, 10);
            expect(weight[4]).to.be.within(-10, 10);

            expect(bias).to.be.within(-10, 10);
        });
    });

    describe('set params', () => {

        it('correctly sets weight and bias', () => {

            const result = Neuron(5);

            result.params = [ 1, 2, 3, 4, 5, 6 ];

            expect(result.weight.length).to.equal(5);
            expect(result.weight[0]).to.equal(2);
            expect(result.weight[1]).to.equal(3);
            expect(result.weight[2]).to.equal(4);
            expect(result.weight[3]).to.equal(5);
            expect(result.weight[4]).to.equal(6);

            expect(result.bias).to.equal(1);
        });
    });

    describe('_weightedSum', () => {

        it('correctly calcs weight * input', () => {

            const n = Neuron(3);
            n.params = [ 10, 1, 2, 3 ];

            const result = n._weightedSum([2, 2, 2]);

            // 2*1 + 2*2 + 2*3 = 12
            expect(result).to.equal(12);
        });
    });

    describe('calc', () => {

        const n = Neuron(3);

        it('correctly calcs output using weight, bias, activity func', () => {

            n.params = [ 10, 1, 2, 3 ];

            const result = n.calc([2, 2, 2]);

            // weighted Sum = 2*1 + 2*2 + 2*3 = 12
            // + bias = 22
            // sigmoid(22) = 0.9999999997210531
            expect(result).to.equal(0.9999999997210531);
        });

        it('stores its output from the last calc', () => {
            expect(n.output).to.equal(0.9999999997210531);
        });
    });

});
