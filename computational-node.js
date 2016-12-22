const _ = require('lodash');

class ComputationalNode {

    constructor (inputNodes = []) {

        this.inputs = [0.0];
        this.output = 0.0;

        this.outputFunction = x => x;
        this.diffed_outputFunction = x => x;

        this.inputNodes = inputNodes.map(n => n.outputNode);
        _.each(this.inputNodes, node => node.outputNodes.push(this));

        this.outputNodes = [];

        this.errorInputs = [0.0];
        this.errorOutput = 0.0;

        this.autoCalc = true;
        this.autoProp = true;
    }

    calc () {
        this.output = this.outputFunction(this.inputs);

        // console.log(this.constructor.name + ' OUTPUT SET TO ' + this.output);
        _.each(this.outputNodes, node => node.input = [this.output, this]);
    }

    backprop() {
        this.errorOutput = this.diffed_outputFunction(this.errorInput)
        this.inputNodes.each(node => node.errorInput = [this.errorOutput, this]);
    }

    set input ([value, sourceNode]) {
        const sourceNodeIndex = _.indexOf(this.inputNodes, sourceNode);
        this.inputs[sourceNodeIndex] = value;
        // console.log(this.constructor.name + ' INPUT ' + sourceNodeIndex + ' SET TO ' + value);

        if (this.autoCalc) {
            this.calc();
        }
    }

    set errInput ([value, sourceNode]) {
        const sourceNodeIndex = _.indexOf(this.outputNodes, sourceNode);
        this.errorInputs[sourceNodeIndex] = value;

        if (this.autoProp) {
            this.backprop();
        }
    }

    get outputNode () {
        return this;
    }
}

const sigmoid = ([x]) => 1.0 / (1.0 + Math.exp(0 - x));
const diffed_sigmoid = x => x * (1.0 - x);

class SigmoidNode extends ComputationalNode {

    constructor (inputNodes = []) {

        if (inputNodes.length > 1) {
            throw new Error('Sigmoid node has only one input');
        }

        super(inputNodes);
        this.outputFunction = sigmoid;
        this.diffed_outputFunction = diffed_sigmoid;
    }
}

class AdditionNode extends ComputationalNode {

    constructor (bias, inputNodes = []) {

        if (inputNodes.length > 1) {
            throw new Error('Addition node has only one input');
        }

        super(inputNodes);
        this.bias = bias;
        this.outputFunction = x => Number(x) + Number(this.bias);
    }
}

class SigmaNode extends ComputationalNode {
    constructor(inputNodes) {
        super(inputNodes);
        this.outputFunction = _.sum;
    }
}

class MultiplyNode extends ComputationalNode {

    constructor (multiplicand, inputNodes = []) {

        if (inputNodes.length > 1) {
            throw new Error('Multply node has only one input');
        }

        super(inputNodes);

        this.multiplicand = multiplicand;
        this.outputFunction = x => x * this.multiplicand;
        this.diffed_outputFunction = x => this.inputs[0];
    }
}

class InputNode {
    constructor(value) {
        this.output = value;
        this.outputNodes = [];
    }

    set input(i) {
        this.output = i;
        _.each(this.outputNodes, node => node.input = [this.output, this]);
    }

    calc() {
    }

    backprop() {
    }

    get outputNode () {
        return this;
    }
}

class PerceptronNode {
    constructor(weights, bias, inputNodes) {

        this.weightNodes = weights.map(
            (weight, inputIndex) => new MultiplyNode(weight, [inputNodes[inputIndex]])
        );

        this.sigmaNode = new SigmaNode(this.weightNodes);
        this.biasNode = new AdditionNode(bias, [this.sigmaNode]);
        this.sigmoidNode = new SigmoidNode([this.biasNode]);

        _.each(this.weightNodes, n => { n.autoCalc = false; })
        this.sigmoidNode.autoProp = false;
    }

    calc() {
        _.each(this.weightNodes, n => n.calc());
        this.sigmaNode.calc();
        this.biasNode.calc();
        this.sigmoidNode.calc();

        return this.output;
    }

    get output () {
        return this.sigmoidNode.output;
    }

    get outputNode () {
        return this.sigmoidNode;
    }
}

const inputs = [0.05, 0.1].map(val => new InputNode(val));

const l1 = [
    new PerceptronNode([0.15, 0.20], 0.35, inputs),
    new PerceptronNode([0.25, 0.30], 0.35, inputs)
];

const l2 = [
    new PerceptronNode([0.40, 0.45], 0.6, l1),
    new PerceptronNode([0.50, 0.55], 0.6, l1)
];

_.each(l1, n => n.calc());
_.each(l2, n => n.calc());

console.log('OUTPUT: ' + l2.map(n => n.output));
