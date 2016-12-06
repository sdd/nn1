const _ = require('lodash');

const activationFunction = input => 1.0 / (1.0 + Math.exp(0 - input));
const generateRandomValue = () => Math.random() * 2 - 1;

module.exports = function Neuron(inputWidth) {

    let output;

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

        get output() {
            return output
        },

        _weightedSum(input) {
            return  _.sum(_.zipWith(input, this.weight, _.multiply));
        },

        calc: function(input) {
            output = activationFunction(this.bias + this._weightedSum(input));
            return output;
        },

        update: function() {
            this.weight = this.newWeight;
            this.bias = this.newBias;
        }
    };
};
