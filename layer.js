const _ = require('lodash');
const Neuron = require('./neuron');

module.exports = function Layer(inputWidth, width) {

    return {
        inputWidth,
        width,
        neurons: _.map(Array(width), () => Neuron(inputWidth)),

        calc (input) {
            return _.map(this.neurons, neuron => neuron.calc(input))
        },

        get params() {
            return _.reduce(this.neurons, (acc, neuron) => [ ...acc, ...neuron.params ], []);
        },

        set params(params) {
            _.reduce(
                this.neurons,
                (acc, neuron) => {
                    neuron.params = acc.splice(0, this.inputWidth + 1);
                    return acc;
                },
                params
            )
        }
    };
};
