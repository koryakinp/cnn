using Cnn.CostFunctions;
using System;
using System.Linq;

namespace Cnn.Layers
{
    internal class OutputLayer : Layer
    {
        private readonly ICostFunction _costFunction;
        public readonly double[] _target;

        public OutputLayer(
            ICostFunction costFunction, 
            int numberOfOutputs, 
            int layerIndex) 
            : base(layerIndex, LayerType.Output)
        {
            _costFunction = costFunction;
            _target = new double[numberOfOutputs];
        }

        public void SetOutput(double[] outputs)
        {
            if(_target.Length != outputs.Length)
            {
                throw new Exception(Consts.NetworkOutputAndTargetDoNotMatch);
            }

            for (int i = 0; i < _target.Length; i++)
            {
                _target[i] = outputs[i];
            }
        }

        public override Value PassBackward(Value value)
        {
            return new SingleValue(value.Single.Select((q, i) => 
                _costFunction.ComputeDeriviative(_target[i], q)).ToArray());
        }

        public override Value PassForward(Value value)
        {
            return value;
        }
    }
}
