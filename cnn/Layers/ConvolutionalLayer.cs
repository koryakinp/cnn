using Cnn.Layers.Abstract;
using Cnn.Layers.Interfaces;
using Cnn.LearningRateAnnealers;
using Cnn.Misc;
using Cnn.WeightInitializers;
using System;
using System.Collections.Generic;

namespace Cnn.Layers
{
    internal class ConvolutionalLayer : FilterLayer, ILearnableLayer
    {
        private readonly IReadOnlyList<Kernel> _kernels;
        private readonly double[,,] _featureMaps;
        private readonly int _kernelSize;
        private readonly int _numberOfKernels;
        private readonly FilterMeta _inputeFm;
        private readonly FilterMeta _outputFm;

        public ConvolutionalLayer(int nk, int ks, int li, FilterMeta ifm, IWeightInitializer wi, LearningRateAnnealerType lrat)
            : base(li, ifm)
        {
            _numberOfKernels = nk;
            _kernelSize = ks;

            List<Kernel> temp = new List<Kernel>();
            for (int i = 0; i < _numberOfKernels; i++)
            {
                var k = new Kernel(ks, ifm.Channels, lrat);
                k.RandomizeWeights(wi);
                temp.Add(k);
            }

            _kernels = new List<Kernel>(temp);
            _inputeFm = ifm;
            _outputFm = GetOutputFilterMeta();
            _featureMaps = new double[_outputFm.Channels, _outputFm.Size, _outputFm.Size];
        }

        public override Value PassBackward(Value value)
        {
            var output = new double[_inputeFm.Channels, _inputeFm.Size, _inputeFm.Size];

            for (int i = 0; i < _numberOfKernels; i++)
            {
                var kernel = value.Multi.GetSlice(i);
                kernel = MatrixProcessor.Pad(kernel, _kernelSize - 1);

                for (int j = 0; j < _inputeFm.Channels; j++)
                {
                    var weight = _kernels[i].Weights.GetSlice(j);
                    weight = MatrixProcessor.Flip(weight);
                    var conv = MatrixProcessor.Convolute(kernel, weight);

                    conv.ForEach((q, ii, jj) => output[j, ii, jj] += q);
                }
            }

            return new MultiValue(output);
        }

        public override Value PassForward(Value value)
        {
            for (int i = 0; i < _kernels.Count; i++)
            {
                MatrixProcessor
                    .Convolute(value.Multi, _kernels[i].Weights)
                    .ForEach((q, j, k) => _featureMaps[i, j, k] = q);
            }

            return new MultiValue(_featureMaps);
        }

        public override FilterMeta GetOutputFilterMeta()
        {
            return new FilterMeta(InputFilterMeta.Size - _kernelSize + 1, _numberOfKernels);
        }

        public void UpdateWeights(double learningRate)
        {
            _kernels.ForEach(q => q.UpdateWeights());
        }

        public void UpdateBiases(double learningRate)
        {
            _kernels.ForEach(q => q.UpdateBias());
        }
    }
}
