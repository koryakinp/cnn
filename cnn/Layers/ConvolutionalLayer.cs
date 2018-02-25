using Cnn.Layers.Abstract;
using Cnn.Layers.Interfaces;
using Cnn.Misc;
using Cnn.WeightInitializers;

namespace Cnn.Layers
{
    internal class ConvolutionalLayer : FilterLayer, ILearnableLayer
    {
        private readonly Kernel[] _kernels;
        private double[][,] _featureMaps;
        private readonly int _kernelSize;
        private readonly int _numberOfKernels;

        public ConvolutionalLayer(
            int numberOfKernels, 
            int kernelSize, 
            int layerIndex, 
            FilterMeta filterMeta,
            IWeightInitializer weightInitializer) 
            : base(layerIndex, LayerType.Convolutional, filterMeta)
        {
            _numberOfKernels = numberOfKernels;
            _kernelSize = kernelSize;
            _kernels = new Kernel[_numberOfKernels];
            for (int i = 0; i < numberOfKernels; i++)
            {
                var kernel = new Kernel(_kernelSize);
                kernel.RandomizeWeights(weightInitializer);
                _kernels[i] = kernel;
            }
        }

        public override Value PassBackward(Value value)
        {
            value = ConvertToMulti(value);

            double[][,] output = new double[_kernels.Length][,];

            for (int i = 0; i < _kernels.Length; i++)
            {
                var kernel = _kernels[i];
                output[i] = new double[_kernelSize, _kernelSize];
                for (int j = 0; j < _featureMaps.Length; j++)
                {
                    var delta = MatrixProcessor.Convolute(_featureMaps[j], value.Multi[i * _kernels.Length + j]);
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
            _featureMaps = value.Multi;
            var output = new double[value.Multi.Length * _kernels.Length][,];
            for (int i = 0; i < _kernels.Length; i++)
            {
                for (int j = 0; j < value.Multi.Length; j++)
                {
                    output[i * value.Multi.Length + j] = 
                        MatrixProcessor.Convolute(value.Multi[j], _kernels[i].Weights);

                    for (int z = 0; z < output[i * value.Multi.Length + j].GetLength(0); z++)
                    {
                        for (int k = 0; k < output[i * value.Multi.Length + j].GetLength(1); k++)
                        {
                            output[i * value.Multi.Length + j][z, k] += _kernels[i].Bias;
                        }
                    }
                }
            }

            return new MultiValue(output);
        }

        public override FilterMeta GetOutputFilterMeta()
        {
            return new FilterMeta(
                InputFilterMeta.Size - _kernelSize + 1, 
                InputFilterMeta.Channels * _numberOfKernels);
        }

        public void UpdateWeights(double learningRate)
        {
            foreach (var kernel in _kernels)
            {
                for (int i = 0; i < kernel.Weights.GetLength(0); i++)
                {
                    for (int j = 0; j < kernel.Weights.GetLength(1); j++)
                    {
                        kernel.Weights[i, j] += kernel.Weights[i, j] * kernel.Gradient[i, j] * learningRate;
                    }
                }
            }
        }

        public void UpdateBiases(double learningRate)
        {
            foreach (var kernel in _kernels)
            {
                double biasDelta = 0;

                for (int i = 0; i < kernel.Gradient.GetLength(0); i++)
                {
                    for (int j = 0; j < kernel.Gradient.GetLength(1); j++)
                    {
                        biasDelta += kernel.Gradient[i, j];
                    }
                }

                kernel.Bias += biasDelta * learningRate;
            }
        }
    }
}
