using cnn;
using Cnn.Activators;
using Cnn.CostFunctions;
using Cnn.Layers;
using Cnn.WeightInitializers;
using System.Collections.Generic;
using System.Linq;

namespace Cnn
{
    public class Network
    {
        internal readonly List<Layer> _layers;
        private readonly ICostFunction _costFunction;
        private readonly IWeightInitializer _weightInitializer;

        public Network(CostFunctionType costFunctionType)
        {
            _costFunction = CostFunctionFactory.Produce(costFunctionType);
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
            _layers.Add(new ConvolutionalLayer(numberOfKernels, kernelSize, _layers.Count + 1));
        }

        public void AddPoolingLayer(int kernelSize)
        {
            _layers.Add(new PoolingLayer(kernelSize, _layers.Count + 1));
        }

        public void AddFullyConnectedLayer(int numberOfNeurons, ActivatorType activatorType)
        {
            if(!_layers.OfType<FullyConnectedLayer>().Any())
            {
                _layers.Add(new BorderLayer(_layers.Count + 1));
            }
            _layers.Add(new FullyConnectedLayer(ActivatorFactory.Produce(activatorType), numberOfNeurons, _layers.Count + 1, _weightInitializer));
        }

        public void AddOutputLayer(int numberOfNeurons, ActivatorType activatorType)
        {
            _layers.Add(new FullyConnectedLayer(ActivatorFactory.Produce(activatorType), numberOfNeurons, _layers.Count + 1, _weightInitializer));
            _layers.Add(new OutputLayer(_costFunction, numberOfNeurons, _layers.Count + 1));
            Configure();
        }

        private void Configure()
        {
            CreateConnections();
        }

        public double TrainModel(double[][,] input, double[] target)
        {
            _layers.OfType<OutputLayer>()
                .First()
                .SetOutput(target);

            var output = PassForward(new MultiValue(input));
            PassBackward(new SingleValue(output));
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

        private void CreateConnections()
        {
            var layers = _layers.OfType<FullyConnectedLayer>().ToList();

            for (int i = 1; i < layers.Count(); i++)
            {
                var prev = layers[i - 1];
                var next = layers[i];

                List<Connection> connections = Utils.CreateConnections(
                    prev.Neurons.Count * next.Neurons.Count,
                    prev.Neurons.Count,
                    _weightInitializer);

                var bwConnections = connections.SplitByStep(next.Neurons.Count);

                next.Neurons.ForEach((q, j) => q.BackwardConnections = bwConnections[j]);
            }
        }
    }
}
