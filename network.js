const _ = require('lodash');
const Layer = require('./layer');

module.exports = function Network(inputWidth, layerWidths) {

    let layers = _.map(layerWidths,
        (layerWidth, layerIndex) => Layer(
            layerIndex === 0 ? inputWidth : layerWidths[layerIndex - 1],
            layerWidth
        )
    );
    let output;

    return {
        inputWidth,
        layerWidths,

        get layers() {
            return layers;
        },

        get output() {
            return output;
        },

        get outputLayer() {
            return _.last(layers);
        },

        get params() {
            return _.flatMap(layers, 'params');
        },

        set params(params) {
            _.reduce(layers, (acc, layer) => {
                layer.params = acc.splice(0, (layer.inputWidth + 1) * layer.width);
                return acc;
            }, params);
        },

        calc (input) {
            layers[-1] = { output: input };
            output = _.flow(_.map(layers, 'calc'))(input);
            return output;
        },

        update: () => _.invokeMap(layers, 'update')
    };
};
