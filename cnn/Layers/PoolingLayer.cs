using Cnn.Misc;
using System;
using System.Linq;

namespace Cnn.Layers
{
    internal class PoolingLayer : Layer
    {
        private readonly int _kernelSize;
        private Coordinate[][] _maxValuesCoordinates;

        public PoolingLayer(int kernelSize, int layerIndex) : base(layerIndex)
        {
            _kernelSize = kernelSize;
        }

        public override Value PassBackward(Value value)
        {
            for (int i = 0; i < value.Multi.Length; i++)
            {
                for (int j = 0; j < value.Multi[i].GetLength(0); j++)
                {
                    for (int k = 0; k < value.Multi[i].GetLength(1); k++)
                    {
                        if(!_maxValuesCoordinates[i].Any(q => q.X == j && q.Y == k))
                        {
                            value.Multi[i][j, k] = 0;
                        }
                    }
                }
            }

            return value;
        }

        public override Value PassForward(Value value)
        {
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
