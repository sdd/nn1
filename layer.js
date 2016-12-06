const _ = require('lodash');
const Neuron = require('./neuron');

module.exports = function Layer(inputWidth, width) {

    const neurons = _.map(Array(width), () => Neuron(inputWidth));
    let output;

    return {
        inputWidth,
        width,

        get neurons() { return neurons; },

        get params() {
            return _.flatMap(neurons, 'params');
        },

        get output() {
            return output;
        },

        set params(params) {
            _.zipWith(
                neurons,
                _.chunk(params, inputWidth + 1),
                (neuron, p) => neuron.params = p
            );
        },

        calc (input) {
            output = _.invokeMap(neurons, 'calc', input);
            return output;
        },

        update: () => _.invokeMap(neurons, 'update')
    };
};
