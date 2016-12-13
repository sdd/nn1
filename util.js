const _ = require('lodash');
const dateFormat = require('dateformat');
const debug = require('debug');
const jsonFile = require('jsonfile');
const d = debug('nn');

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

const avgVectorDistance = vectorPairList => _.mean(vectorPairList.map(vectorDistance));

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

const backPropagateDetailed = (network, input, expected, learningRate) => {

    // Massive props to Matt Mazur, whose blog post has helped me greatly
    // https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

    // this is also handy: https://www.cs.swarthmore.edu/~meeden/cs81/s10/BackPropDeriv.pdf

    // propagate forward
    network.calc(input);
    d('network output: ' + JSON.stringify(_.map(network.layers, 'output')));

    // calc total error
    const E_total = _.zipWith(expected, network.output, scalarOutputError);
    d('E_total: ' + E_total);

    // calculate the output error gradient for the output layer
    network.outputLayer.del_E_total_by_del_out = _.zipWith(network.output, expected, _.subtract);
    d('∂_E_total_by_∂_out[last] = ' + network.outputLayer.del_E_total_by_del_out);

    // backward propagate error gradients
    _.eachRight(network.layers, (layer, l) => {

        // calculate the output error gradient for this layer
        if (l !== network.layers.length - 1) {
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

            d('delta_j:' + delta_j);

            neuron.del_E_total_by_del_weight = _.zipWith(
                neuron.weight, prevLayerOutput,
                (weight, prevLayerOut) => {

                    const del_net_i_by_del_weight_j = prevLayerOut;
                    const del_E_total_by_del_weight_j = delta_j * del_net_i_by_del_weight_j;

                    d('∂_net_i_by_∂_weight_j = ' + del_net_i_by_del_weight_j);
                    d('∂_E_total_by_∂_weight_j:' + del_E_total_by_del_weight_j);

                    return del_E_total_by_del_weight_j;
                }
            );

            // bias error gradient calculation from this SO answer:
            // http://stackoverflow.com/a/13342725
            neuron.biasErrGradient = delta_j;

            // now that we know the error gradient for this weight we can learn!
            // set the new weight to the old one minus the learning rate times the gradient,
            // to descend down the error surface by a distance of learning_rate. Keep the
            // old value in place for now as preceeding layers will need it to calculate
            // their output error.
            neuron.newWeight = _.zipWith(
                neuron.del_E_total_by_del_weight, neuron.weight,
                (gradient, weight) => weight - (learningRate * gradient)
            );

            // also see http://stackoverflow.com/a/13342725
            neuron.newBias = neuron.bias - (learningRate * neuron.biasErrGradient);

            d('new weight: [' + neuron.newWeight + ']');
            d('new bias: ' + neuron.newBias);
        });
    });

    // update all weights and biases with new values
    network.update();
};

const backPropagate = (network, lesson, hyperParams) => {

    const {
        learningRate
    } = hyperParams;

    const { input, output } = lesson;

    // propagate forward
    // console.time('forward');
    network.calc(input);
    // console.timeEnd('forward');

    // backward propagate error gradients, starting at the output
    _.eachRight(network.layers, (layer, layerIndex) => {
        console.time('layer ' + layerIndex);

        const nextLayer = network.layers[layerIndex + 1];
        const layerInput = network.layers[layerIndex - 1].output;

        if (!nextLayer) {

            // calculate output error gradient for output layer
            layer.del_E_total_by_del_out = _.zipWith(network.output, output, _.subtract);
        } else {

            // calculate output error gradient for earlier layers
            layer.del_E_total_by_del_out = _.times(layer.neurons.length, i => _.sum(
                _.zipWith(
                    nextLayer.delta, nextLayer.neurons,
                    (delta, { weight }) => delta * weight[i]
                )
            ));
        }

        layer.del_out_by_del_net = _.map(layer.output, diffedSigmoid);

        layer.delta = _.zipWith(layer.del_E_total_by_del_out, layer.del_out_by_del_net, _.multiply);

        // console.time('weights ' + layerIndex);

        /*

        // slow but nice version
        _.zipWith(layer.neurons, layer.delta, (neuron, delta) => {

            // console.time('del_weight');
            neuron.del_E_total_by_del_weight = _.map(layerInput, _.partial(_.multiply, delta));
            // console.timeEnd('del_weight');

            // console.time('new_weight');
            neuron.newWeight = _.zipWith(
                neuron.del_E_total_by_del_weight, neuron.weight,
                (gradient, weight) => weight - (learningRate * gradient)
            );
            // console.timeEnd('new_weight');

            neuron.newBias = neuron.bias - (learningRate * delta);
        });

        /*/

        // quick tuned version
        let neurons = layer.neurons,
            neuron,
            weight,
            len_n = neurons.length,
            len_i = layerInput.length,
            delta = layer.delta,
            lr_delta;

        for(let i = 0; i < len_n; i++) {

            neuron = neurons[i];
            weight = neuron.weight;
            lr_delta = learningRate * delta[i];

            for(let j = 0; j < len_i; j++) {
                neuron.newWeight[j] = weight[j] - (lr_delta * layerInput[j]);
                neuron.newBias = neuron.bias - lr_delta;
            }
        }
        //*/

        // console.timeEnd('weights ' + layerIndex);

        console.timeEnd('layer ' + layerIndex);
    });

    // update all weights and biases with new values
    network.update();
};

const backPropagateTuned = (network, lesson, hyperParams) => {

    const {
        learningRate
    } = hyperParams;

    const { input, output } = lesson;

    // propagate forward
    // console.time('forward');
    network.calc(input);
    // console.timeEnd('forward');

    // backward propagate error gradients, starting at the output
    _.eachRight(network.layers, (layer, layerIndex) => {

        console.time('layer ' + layerIndex);
        const nextLayer = network.layers[layerIndex + 1];
        const layerInput = network.layers[layerIndex - 1].output;

        if (!nextLayer) {

            // calculate output error gradient for output layer
            layer.del_E_total_by_del_out = _.zipWith(network.output, output, _.subtract);
        } else {

            // calculate output error gradient for earlier layers
            layer.del_E_total_by_del_out = _.times(layer.neurons.length, i => _.sum(
                _.map(nextLayer.neurons, n => n.delta * n.weight[i])
            ));
        }

        console.log('del_E_total_by_del_out old: ' + layer.del_E_total_by_del_out);

        // quick tuned version
        let neurons = layer.neurons,
            nextNeurons = nextLayer && nextLayer.neurons,
            len_n = neurons.length,
            len_next = nextNeurons && nextNeurons.length,
            len_i = layerInput.length,
            delta = layer.delta,
            o = layer.output,
            neuron,
            nextNeuron,
            weight,
            del_E_total_by_del_out,
            lr_delta,
            j;

        for(let i = 0; i < len_n; i++) {

            neuron = neurons[i];

            del_E_total_by_del_out = 0;
            for(j = 0; j < len_next; j++) {
                nextNeuron = nextNeurons[j];
                del_E_total_by_del_out += (nextNeuron.delta * nextNeuron.weight[i]);
            }

            console.log('del_E_total_by_del_out new: ' + del_E_total_by_del_out);

            neuron.delta = delta = del_E_total_by_del_out * diffedSigmoid(o[i]);
            lr_delta = learningRate * delta;
            weight = neuron.weight;

            for(let j = 0; j < len_i; j++) {
                neuron.newWeight[j] = weight[j] - (lr_delta * layerInput[j]);
                neuron.newBias = neuron.bias - lr_delta;
            }
        }

        console.timeEnd('layer ' + layerIndex);
    });

    // update all weights and biases with new values
    network.update();
};

const indexOfMax = arr => _.indexOf(arr, _.max(arr));

const calcAccuracy = (network, testSet) => {
    return _.sum(
        testSet.map(test => {
            const expectedClassification = indexOfMax(test.output);
            const actualClassification = indexOfMax(network.calc(test.input));

            return expectedClassification === actualClassification ? 1 : 0;
        })
    ) / testSet.length;
};

const decimalToPercent = (val, precision = 2) => `${ (val * 100).toFixed(precision) }%`;

const statTracker = (tag = 'default') => {

    const data = [];
    let runStartTime = process.uptime();
    let splitTime = runStartTime;

    const log = (datum) => {
        data.push({ splitTime: process.uptime() - splitTime, data: datum });
        splitTime = process.uptime();
    };

    const dump = () => {
        const dumpDateTime = dateFormat(new Date(), 'yyyy-mm-dd-HH-MM');

        jsonFile.writeFileSync(
            `./log-${ tag }-${ dumpDateTime }.json`,
            {
                startTime: runStartTime,
                endTime: process.uptime(),
                data
            }
        );
    };

    return { dump, log };
};

const progressTracker = (updateInterval = 1000) => {

    let state = {};

    const showState = function() {
        console.log(state);
        process.stdout.write('' + JSON.stringify(state) + '\r');
    };

    setInterval(showState, updateInterval);

    return update => {
        state = { ...state, update };
        console.log(state);
    };
};

module.exports = {
    avgVectorDistance,
    backPropagate,
    backPropagateDetailed,
    backPropagateTuned,
    calcAccuracy,
    checkLength,
    computeNumericalGradient,
    decimalToPercent,
    evaluateCost,
    indexOfMax,
    statTracker,
    vectorDistance,
    progressTracker
};
