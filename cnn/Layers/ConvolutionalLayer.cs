using System;
using System.Collections.Generic;
using System.Linq;

namespace Cnn.Layers
{
    internal class ConvolutionalLayer : Layer
    {
        private readonly Kernel[] _kernels;
        private readonly int _kernelSize;

        public ConvolutionalLayer(int numberOfKernels, int kernelSize, int layerIndex) 
            : base(layerIndex)
        {
            _kernelSize = kernelSize;
            _kernels = new Kernel[numberOfKernels];
            for (int i = 0; i < numberOfKernels; i++)
            {
                var kernel = new Kernel(kernelSize);
                kernel.RandomizeWeights();
                _kernels[i] = kernel;
            }
        }

        public ConvolutionalLayer(
            int numberOfKernels, 
            int kernelSize, 
            int layerIndex, 
            double[][,] kernels, 
            double[][,] weights) : base(layerIndex)
        {
            _kernels = new Kernel[numberOfKernels];
            for (int i = 0; i < numberOfKernels; i++)
            {
                _kernels[i] = new Kernel(kernels[i], weights[i]);
            }
        }

        public override Value PassBackward(Value value)
        {
            double[][,] output = new double[_kernels.Length][,];

            for (int i = 0; i < _kernels.Length; i++)
            {
                var kernel = _kernels[i];
                output[i] = new double[_kernelSize, _kernelSize];
                for (int j = 0; j < kernel.FeatureMaps.Length; j++)
                {
                    var delta = MatrixProcessor
                        .Convolute(kernel.FeatureMaps[j], value.Multi[j]);
                    output[i] = MatrixProcessor.Add(output[i], delta);
                }

                for (int q = 0; q < kernel.Gradient.GetLength(0); q++)
                {
                    for (int w = 0; w < kernel.Gradient.GetLength(1); w++)
                    {
                        kernel.Gradient[q, w] = output[i][q,w];
                    }
                }
            }

            return new MultiValue(output);
        }

        public override Value PassForward(Value value)
        {
            for (int i = 0; i < _kernels.Length; i++)
            {
                _kernels[i].FeatureMaps = new double[value.Multi.Length][,];
                for (int j = 0; j < value.Multi.Length; j++)
                {
                    _kernels[i].FeatureMaps[j] = MatrixProcessor
                        .Convolute(value.Multi[j], _kernels[i].Weights);
                }
            }

            return new MultiValue(_kernels.SelectMany(q => q.FeatureMaps).ToArray());
        }
    }
}
