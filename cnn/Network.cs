using Cnn.Activators;
using Cnn.CostFunctions;
using Cnn.Layers;
using Cnn.WeightInitializers;
using System.Collections.Generic;
using System.Linq;
using Cnn.Layers.Abstract;
using Cnn.Misc;

namespace Cnn
{
    public class Network
    {
        internal readonly List<Layer> _layers;
        private readonly ICostFunction _costFunction;
        private readonly IWeightInitializer _weightInitializer;
        private readonly NetworkConfiguration _networkConfig;

        public Network(NetworkConfiguration networkConfig)
        {
            _networkConfig = networkConfig;
            _costFunction = CostFunctionFactory.Produce(_networkConfig.CostFunctionType);
            _layers = new List<Layer>();
            _weightInitializer = new WeightInitializer();
        }

        internal Network(CostFunctionType costFunctionType, IWeightInitializer weightInitializer) 
        {
            _weightInitializer = weightInitializer;
            _costFunction = CostFunctionFactory.Produce(costFunctionType);
            _layers = new List<Layer>();
        }

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
            if(_layers.Any())
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

        public double TrainModel(double[][,] input, double[] target)
        {
            var output = PassForward(new MultiValue(input));

            double[] errors = output
                .Select((q, i) => _costFunction.ComputeDeriviative(target[i], q))
                .ToArray();

            PassBackward(new SingleValue(errors));

            return output.Select((q,i) => _costFunction.ComputeValue(target[i], q)).Sum();
        }

        private void PassBackward(Value value)
        {
            _layers.OrderByDescending(q => q.LayerIndex).ForEach(q => value = q.PassBackward(value));
        }

        private double[] PassForward(Value value)
        {
            _layers.ForEach(q => value = q.PassForward(value));
            return value.Single;
        }
    }
}
