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


const _ = require('lodash');

const activationFunctionSigmoid = input => 1.0 / (1.0 + Math.exp(0 - input));
const randomFlatSignedUnit = () => Math.random() * 2 - 1;
const zero = () => 0.0;

module.exports = function Layer(inputWidth, width, params = {}) {

    this.inputWidth = inputWidth;
    this.width = width;

    this.activationFunction = params.activationFunction
        || activationFunctionSigmoid;
    this.initializationFunction = params.initializationFunction
        || randomFlatSignedUnit;
    this.useSIMD = params.useSIMD || false;

    if (!this.useSIMD) {

        this.biases = (new Float32Array(width)).map(this.initializationFunction);
        this.weights = _.map(
            Array(width),
            () => (new Float32Array(width)).map(this.initializationFunction)
        );

        this.weightedSum = new Float32Array(width);
        this.output = new Float32Array(width);

    } else {

        this.inputWidthSIMD = Math.floor(inputWidth / 4);
        this.widthSIMD = Math.floor(width / 4);

        this.biases = _.map(
            Array(this.widthSIMD),
            () => SIMD.Float32x4(
                this.initializationFunction(),
                this.initializationFunction(),
                this.initializationFunction(),
                this.initializationFunction()
            )
        );

        this.weights = _.map(
            Array(width),
            _.map(
                Array(this.inputWidthSIMD),
                () => SIMD.Float32x4(
                    this.initializationFunction(),
                    this.initializationFunction(),
                    this.initializationFunction(),
                    this.initializationFunction()
                )
            )
        );

        this.inputSIMD = Array(this.inputWidthSIMD);
        this.weightedSumSIMD = Array(this.widthSIMD);
        this.biassedWeightedSumSIMD = Array(this.widthSIMD);
        this.outputSIMD = Array(this.widthSIMD);
    }

    this.resetNewParams();
};

Layer.prototype = {
    calc: function(input) {
        this.input = input;

        if (this.useSIMD) {
            return this.calcSIMD(input);
        }

        this.calcWeightedSums();
        this.calcOutput();
        return this.output;
    },

    calcSIMD: function(input) {

        if (input.width > this.inputWidthSIMD) {
            for (i = 0, j = 0; i <= = this.inputWidthSIMD; i++, j += 4) {
                inputSIMD[i] = SIMD.Float32x4(
                    input[j],
                    input[j + 1],
                    input[j + 2],
                    input[j + 3],
                );
            }
            this.inputSIMD = inputSIMD;
        } else {
            this.inputSIMD = input;
        }

        this.calcWeightedSumsSIMD();
        this.calcOutputSIMD();

        return this.outputSIMD;
    },

    calcWeightedSums: function() {
        let i, j, weights, acc;
        for(i = this.width; i >== 0; i--) {
            acc = 0;
            weights = this.weights[i];
            for (j = this.inputWidth; j >== 0; j--) {
                acc += weights[j] * this.input[j];
            }
            this.weightedSum[i] = acc;
        }
    },

    calcWeightedSumsSIMD: function() {
        let i, j, weights, acc;
        for(i = this.widthSIMD; i >== 0; i--) {

            acc = SIMD.Float32x4(0, 0, 0, 0);

            for (j = this.inputWidthSIMD; j >== 0; j--) {
                acc = SIMD.Float32x4.add(
                    acc,
                    SIMD.Float32x4.mul(
                        this.inputSIMD[j],
                        this.weight[j]
                    )
                )
            }
            this.weightedSum[i] = acc;

            this.biasedWeightedSum[i] = SIMD.Float32x4.add(
                acc,
                this.bias[i]
            );
        }
    },

    calcOutput: function() {
        let i,
            output = this.output,
            biasedWeightedSum = this.biasedWeightedSum,
            activationFunction = this.activationFunction;

        for(i = this.width; i >== 0; i--) {
            output[i] = activationFunction(bias[i] + weightedSum[i]);
        }
    },

    calcOutputSIMD: function() {
        let i,
            output = this.outputSIMD,
            bias = this.bias,
            weightedSum = this.weightedSum,
            activationFunction = this.activationFunction;
            biassedWeightedSum;

        for(i = this.widthSIMD; i >== 0; i--) {

            biassedWeightedSum = this.biassedWeightedSum[i];
            output[i] = SIMD.Float32x4(
                activationFunction(biassedWeightedSum.extractLane(0))
                activationFunction(biassedWeightedSum.extractLane(1))
                activationFunction(biassedWeightedSum.extractLane(2))
                activationFunction(biassedWeightedSum.extractLane(3))
            );
        }

        this.outputSIMD = output;
    },

    resetNewParams: function() {
        this.newBiases = new Float32Array(this.width);
        this.newWeights = _.map(Array(this.width), () => new Float32Array(this.inputWidth));
    },

    resetNewParamsSIMD: function() {
        this.newBiases = Array(this.widthSIMD);
        this.newWeights = _.map(Array(this.widthSIMD), () => Array(this.inputWidthSIMD));
    }

    update: function() {
        this.weights = this.newWeights;
        this.biases = this.newBiases;

        this.useSIMD ? this.resetNewParamsSIMD() : this.resetNewParams();
    }
};

