const _ = require('lodash');

const activationFunction = input => 1.0 / (1.0 + Math.exp(0 - input));
const generateRandomValue = () => Math.random() * 2 - 1;

module.exports = function Neuron(inputWidth) {

    return {
        bias: generateRandomValue(),
        weight: _.map(Array(inputWidth), generateRandomValue),

        get params() {
            return [ this.bias, ...this.weight ]
        },

        set params (params) {
            const [ bias, ...weight ] = params;
            this.bias = bias;
            this.weight = weight;
        },

        _weightedSum(inputs) {
            return  _.reduce(
                inputs,
                (sum, input, idx) => sum + (input * this.weight[idx]),
                0
            );
        },

        calc: function(inputs) {

            if (inputs.length !== this.weight.length) {
                throw new Error(`Bad input dimension (expected ${this.weight.length} but got ${input.length})`);
            }

            return activationFunction(this.bias + this._weightedSum(inputs));
        }
    };
};
