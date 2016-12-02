const _ = require('lodash');
const Neuron = require('./neuron');

module.exports = function Layer(inputWidth, width) {

    return {
        inputWidth,
        width,
        neurons: _.map(Array(width), () => Neuron(inputWidth)),

        calc (input) {
            return _.invokeMap(this.neurons, 'calc', input);
        },

        get params() {
            return _.flatMap(this.neurons, 'params');
        },

        set params(params) {
            _.zipWith(
                this.neurons,
                _.chunk(params, this.inputWidth + 1)),
                (neuron, p) => neuron.params = p
            );
        }
    };
};
