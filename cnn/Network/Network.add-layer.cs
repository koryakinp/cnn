﻿using Cnn.Activators;
using Cnn.Layers;
using Cnn.Layers.Abstract;
using Cnn.Misc;
using Cnn.WeightInitializers;
using System.Linq;

namespace Cnn
{
    public partial class Network
    {
        public void AddConvolutionalLayer(int numberOfKernels, int kernelSize)
        {
            if (_layers.Any())
            {
                _layers.Add(new ConvolutionalLayer(
                    numberOfKernels,
                    kernelSize,
                    _layers.Last().LayerIndex + 1,
                    _layers.OfType<FilterLayer>().Last().GetOutputFilterMeta(),
                    new WeightInitializer()));
            }
            else
            {
                _layers.Add(new ConvolutionalLayer(
                    numberOfKernels,
                    kernelSize,
                    1,
                    new FilterMeta(_networkConfig.InputDimenision, _networkConfig.InputChannels),
                    new WeightInitializer()));
            }
        }

        public void AddPoolingLayer(int kernelSize)
        {
            if (_layers.Any())
            {
                _layers.Add(new PoolingLayer(
                    kernelSize,
                    _layers.Last().LayerIndex + 1,
                    _layers.OfType<FilterLayer>().Last().GetOutputFilterMeta()));
            }
            else
            {
                _layers.Add(new PoolingLayer(
                    kernelSize,
                    1,
                    new FilterMeta(_networkConfig.InputDimenision, _networkConfig.InputChannels)));
            }
        }

        public void AddDetectorLayer(ActivatorType activatorType)
        {
            if (_layers.Any())
            {
                _layers.Add(new DetectorLayer(
                    _layers.Last().LayerIndex + 1,
                    ActivatorFactory.Produce(activatorType),
                    _layers.OfType<FilterLayer>().Last().GetOutputFilterMeta()));
            }
            else
            {
                _layers.Add(new DetectorLayer(
                    1,
                    ActivatorFactory.Produce(activatorType),
                    new FilterMeta(_networkConfig.InputDimenision, _networkConfig.InputChannels)));
            }
        }

        public void AddFullyConnectedLayer(int numberOfNeurons, ActivatorType activatorType)
        {
            _layers.Add(new FullyConnectedLayer(
                ActivatorFactory.Produce(activatorType),
                numberOfNeurons,
                _layers.Last().GetNumberOfOutputValues(),
                _layers.Last().LayerIndex + 1,
                _weightInitializer));
        }
    }
}