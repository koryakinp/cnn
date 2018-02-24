using Cnn.Layers.Abstract;
using Cnn.Misc;
using System;

namespace Cnn.Layers
{
    internal class PoolingLayer : FilterLayer
    {
        private readonly int _kernelSize;
        private readonly Coordinate[][] _maxValuesCoordinates;
        private readonly double[][,] _inputMaps;

        public PoolingLayer(int kernelSize, int layerIndex, FilterMeta filterMeta) 
            : base(layerIndex, LayerType.Pooling, filterMeta)
        {
            _maxValuesCoordinates = new Coordinate[filterMeta.Channels][];
            _inputMaps = new double[filterMeta.Channels][,];
            _kernelSize = kernelSize;
        }

        public override FilterMeta GetOutputFilterMeta()
        {
            int size;
            if(InputFilterMeta.Size % _kernelSize == 0)
            {
                size = InputFilterMeta.Size / _kernelSize;
            }
            else
            {
                size = InputFilterMeta.Size / _kernelSize + 1;
            }

            return new FilterMeta(size, InputFilterMeta.Channels);
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
            for (int i = 0; i < value.Multi.Length; i++)
            {
                _inputMaps[i] = new double[value.Multi[i].GetLength(0), value.Multi[i].GetLength(1)];
            }

            var output = new MultiValue(new double[value.Multi.Length][,]);

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
