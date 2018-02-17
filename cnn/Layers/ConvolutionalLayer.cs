using System;
using System.Collections.Generic;
using System.Linq;

namespace Cnn.Layers
{
    internal class ConvolutionalLayer : Layer
    {
        private readonly Kernel[] _kernels;
        private double[][,] _featureMaps { get; set; }

        public ConvolutionalLayer(int numberOfKernels, int kernelSize, int layerIndex) : base(layerIndex)
        {
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
            _featureMaps = new double[numberOfKernels][,];
            _kernels = new Kernel[numberOfKernels];
            for (int i = 0; i < numberOfKernels; i++)
            {
                _kernels[i] = new Kernel(kernels[i], weights[i]);
            }
        }

        public override Value PassBackward(Value value)
        {
            throw new NotImplementedException();
        }

        public override Value PassForward(Value value)
        {
            _featureMaps = new double[_kernels.Length * value.Multi.Length][,];

            for (int i = 0; i < _kernels.Length; i++)
            {
                for (int j = 0; j < value.Multi.Length; j++)
                {
                    _featureMaps[i * j + j] = MatrixProcessor
                        .Convolute(value.Multi[j], _kernels[i].Weights);
                }
            }

            return new MultiValue(_featureMaps);
        }
    }
}
