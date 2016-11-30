const _ = require('lodash');
const debug = require('debug');
const d = debug('nn:debug');

const checkLength = target => (a, b) => {

    // accept args in the form  (a, b) or ([a, b])
    if (b.length === undefined) { [a, b] = a; }

    if (a.length !== b.length) {
        throw new Error(`vector dimensions do not match (${a.length} != ${b.length})`);
    }
    return target(a, b);
};

const scalarOutputError = (a, b) => 0.5 * (a - b) ** 2;

const vectorDistance = checkLength((a, b) => {

    const result = Math.sqrt(
        _.sum(_.zip(a, b).map(([i, j]) => (i - j) ** 2))
    );

    //console.log('dist from ' + a + ' to ' + b + ' = ' + result);

    return result;
});

const avgVectorDistance = vectorPairList =>
    _.mean(vectorPairList.map(vectorDistance))
;

const evaluateCost = (network, trainingSet) => {

    const errorVectorPairs = trainingSet
        .map(({ input, output }) => [network.calc(input), output]);

    // console.log('errorVectorPairs: ' + JSON.stringify(errorVectorPairs));

    return avgVectorDistance(errorVectorPairs);
};

const computeNumericalGradient = (func, input, h = 0.00001) => {

    const f = func(input);

    const f_nabla = input.map((element, i) => {

        const f_prime = func(input.map(
            (coord, j) => i === j
                ? coord + h
                : coord
        ));

        return (f_prime[i] - f[i]) / h;
    });

    return f_nabla;
};

const diffedSigmoid = x => x * (1 - x);

const backPropagate = (network, input, expected, learningRate) => {

    // Massive props to Matt Mazur, whose blog post has helped me greatly
    // https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

    // this is also handy: https://www.cs.swarthmore.edu/~meeden/cs81/s10/BackPropDeriv.pdf

    // propagate forward
    network.layers.reduce(
        (acc, layer) => {
            layer.output = layer.calc(acc);
            return layer.output;
        },
        input
    );

    d('network output: ' + JSON.stringify(
        _.map(network.layers, 'output')
    ));

    // calc total error
    const E_total = _.zip(expected, _.last(network.layers).output).map(
        values => scalarOutputError(...values)
    );

    d('E_total: ' + E_total);

    // propagate backward
    _.eachRight(network.layers, (layer, l) => {

        // calculate the output error gradient for this layer
        if (l === network.layers.length - 1) {

            // output layer
            layer.del_E_total_by_del_out = layer.output.map(
                (out, i) => out - expected[i]
            );

        } else {

            // hidden layers
            layer.del_E_total_by_del_out = layer.neurons.map(
                (neuron, i) => {

                    // ∂_E_total          ∂_E_out_l_plus_1
                    // --------- = sigma  ----------------
                    //  ∂_out_i           ∂_out_i

                    const nextLayer = network.layers[l + 1];

                    const del_E_total_by_del_out_i = _.sum(
                        nextLayer.neurons.map((nextLayerNeuron, i2) => {

                            const del_e_i_by_del_net_i = nextLayer.del_E_total_by_del_out[i2]
                                * nextLayerNeuron.del_out_by_del_net;

                            d('∂_E_i_by_∂_net_i = ∂_E_out_l_plus_1/∂_out_i = '
                                + nextLayer.del_E_total_by_del_out[i2]
                                + ' * '
                                + nextLayerNeuron.del_out_by_del_net
                                + ' = ' + del_e_i_by_del_net_i
                            );

                            const del_net_i_by_del_out_i = nextLayerNeuron.weight[i];

                            const del_E_i_by_del_out_i = del_e_i_by_del_net_i * del_net_i_by_del_out_i;

                            d('∂_E_i_by_∂_out_i = ∂_E_i_by_∂_net_i * ∂_net_i_by_∂_out_i = '
                                + del_e_i_by_del_net_i + ' * ' + del_net_i_by_del_out_i
                                + ' = ' + del_E_i_by_del_out_i
                            );

                            return del_E_i_by_del_out_i;
                        })
                    );

                    d('∂_E_total_by_∂_out_i = ' + del_E_total_by_del_out_i);
                    return del_E_total_by_del_out_i;
                }
            );
        }

        d('layer.∂_E_total_by_∂_out = ' + layer.del_E_total_by_del_out);

        const prevLayerOutput = l === 0 ? input : network.layers[l-1].output;

        // use this layer's output error gradient to calculate the error gradient
        // for each neuron in this layer's weights
        _.each(layer.neurons, (neuron, i) => {

            // ∂_E_total    ∂_E_total   ∂_out_i    ∂_net_i
            // ---------- = --------- * ------- * ----------
            // ∂_weight_j    ∂_out_i    ∂_net_i   ∂_weight_j

            neuron.del_out_by_del_net = diffedSigmoid(layer.output[i]);
            d('neuron.∂_out_by_∂_net = ' + neuron.del_out_by_del_net);

            // see https://en.wikipedia.org/wiki/Delta_rule
            const delta_j = layer.del_E_total_by_del_out[i]
                * neuron.del_out_by_del_net;

            neuron.weightErrGradient = _.map(neuron.weight, (weight, j) => {

                const del_net_i_by_del_weight_j = prevLayerOutput[j];
                d('∂_net_i_by_∂_weight_j = ' + del_net_i_by_del_weight_j);

                    const del_E_total_by_del_weight_j = delta_j * del_net_i_by_del_weight_j;

                    d('∂_E_total_by_∂_weight_j:' + del_E_total_by_del_weight_j);

                    return del_E_total_by_del_weight_j;
            });

            // bias error gradient calculation from this SO answer:
            // http://stackoverflow.com/a/13342725
            neuron.biasErrGradient = delta_j;

            // now that we know the error gradient for this weight we can learn!
            // set the new weight to the old one minus the learning rate times the gradient,
            // to descend down the error surface by a distance of learning_rate. Keep the
            // old value in place for now as preceeding layers will need it to calculate
            // their output error.
            neuron.newWeight = neuron.weightErrGradient.map(
                (gradient, j) => neuron.weight[j] - (learningRate * gradient)
            );

            // also see http://stackoverflow.com/a/13342725
            neuron.newBias = neuron.bias - (learningRate * neuron.biasErrGradient);

            d('new weight: [' + neuron.newWeight + ']');
            d('new bias: ' + neuron.newBias);
        });
    });

    // update all weights and biases with new values
    _.each(network.layers,
        layer => _.each(layer.neurons, neuron => {
            neuron.weight = neuron.newWeight;
            neuron.bias = neuron.newBias;
        })
    );
};

module.exports = {
    avgVectorDistance,
    backPropagate,
    checkLength,
    computeNumericalGradient,
    evaluateCost,
    vectorDistance
};
