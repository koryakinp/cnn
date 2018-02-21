using Cnn.Misc;
using System;
using System.Linq;

namespace Cnn.Layers
{
    internal class PoolingLayer : Layer
    {
        private readonly int _kernelSize;
        private Coordinate[][] _maxValuesCoordinates;
        private double[][,] _inputMaps;

        public PoolingLayer(int kernelSize, int layerIndex) : base(layerIndex, LayerType.Pooling)
        {
            _kernelSize = kernelSize;
        }

        public override Value PassBackward(Value value)
        {
            var output = new double[value.Multi.Length][,];

            for (int i = 0; i < value.Multi.Length; i++)
            {
                output[i] = MatrixProcessor.ReverseMaxPool(
                    value.Multi[i], 
                    _kernelSize, 
                    _inputMaps[0].GetLength(0), 
                    _maxValuesCoordinates[i]);
            }

            return new MultiValue(output);
        }

        public override Value PassForward(Value value)
        {
            _inputMaps = new double[value.Multi.Length][,];
            for (int i = 0; i < value.Multi.Length; i++)
            {
                _inputMaps[i] = new double[value.Multi[i].GetLength(0), value.Multi[i].GetLength(1)];
            }

            var output = new MultiValue(new double[value.Multi.Length][,]);
            _maxValuesCoordinates = new Coordinate[value.Multi.Length][];

            for (int i = 0; i < value.Multi.Length; i++)
            {
                if (value.Multi[i].GetLength(0) < _kernelSize || value.Multi[i].GetLength(1) < _kernelSize)
                {
                    throw new InvalidOperationException(Consts.FeatureMapMustBeBiggerThanKernel);
                }

                var res = MatrixProcessor.MaxPool(value.Multi[i], _kernelSize);
                output.Multi[i] = res.Values;
                _maxValuesCoordinates[i] = res.MaxCoordinates;
            }

            return output;
        }
    }
}
