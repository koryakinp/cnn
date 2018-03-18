using Cnn.Layers.Abstract;
using Cnn.Misc;
using System;

namespace Cnn.Layers
{
    internal class PoolingLayer : FilterLayer
    {
        private readonly int _kernelSize;
        private readonly bool[,,] _maxValues;
        private readonly double[,,] _inputMaps;

        public PoolingLayer(int kernelSize, int layerIndex, FilterMeta filterMeta) 
            : base(layerIndex, filterMeta)
        {
            _maxValues = new bool[filterMeta.Channels, filterMeta.Size, filterMeta.Size];
            _inputMaps = new double[filterMeta.Channels, filterMeta.Size, filterMeta.Size];
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
            var output = MatrixProcessor.ReverseMaxPool(value.Multi, _maxValues, _kernelSize);
            return new MultiValue(output);
        }

        public override Value PassForward(Value value)
        {
            var res = MatrixProcessor.MaxPool(value.Multi, _kernelSize);
            _maxValues.ForEach((k, i, j) => _maxValues[k, i, j] = res.Item2[k, i, j]);
            return new MultiValue(res.Item1);
        }
    }
}
