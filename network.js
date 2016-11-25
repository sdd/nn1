const _ = require('lodash');
const Layer = require('./layer');

module.exports = function Network(inputWidth, layerWidths) {

    return {
        inputWidth,
        layerWidths,
        layers: _.map(layerWidths,
            (layerWidth, layerIndex) => Layer(
                layerIndex === 0
                    ? inputWidth
                    : layerWidths[layerIndex - 1],
                layerWidth
            )
        ),

        calc (inputs) {
            return _.reduce(this.layers, (acc, layer) => layer.calc(acc), inputs)
        },

        get params() {
            return _.reduce(this.layers, (acc, layer) => [ ...acc, ...layer.params ], []);
        },

        set params(params) {
            _.reduce(
                this.layers,
                (acc, layer) => {
                    layer.params = acc.splice(0, (layer.inputWidth + 1) * layer.width);
                    return acc;
                },
                params
            )
        }
    };
};
