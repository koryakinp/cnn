using Cnn.CostFunctions;
using Cnn.WeightInitializers;
using System.Collections.Generic;
using System.Linq;
using Cnn.Layers.Abstract;
using Cnn.Misc;
using Cnn.Layers.Interfaces;

namespace Cnn
{
    public partial class Network
    {
        private readonly List<Layer> _layers;
        private readonly ICostFunction _costFunction;
        private readonly IWeightInitializer _weightInitializer;
        private readonly NetworkConfiguration _networkConfig;
        private int epoch = 1;

        public Network(NetworkConfiguration networkConfig)
        {
            _networkConfig = networkConfig;
            _costFunction = CostFunctionFactory.Produce(_networkConfig.CostFunctionType);
            _layers = new List<Layer>();
            _weightInitializer = new WeightInitializer();
        }

        public double TrainModel(double[,,] input, double[] target)
        {
            var output = PassForward(new MultiValue(input));

            double[] errors = ComputeCost(output, target);

            PassBackward(new SingleValue(errors));

            Learn(_networkConfig.Decayer.LearningRate);

            _networkConfig.Decayer.Decay(epoch++);

            return ComputeError(output, target);
        }

        private double[] ComputeCost(double[] output, double[] target)
        {
            return output
                .Select((q, i) => _costFunction.ComputeDeriviative(target[i], q))
                .ToArray();
        }

        private double ComputeError(double[] output, double[] target)
        {
            return output
                .Select((q, i) => _costFunction.ComputeValue(target[i], q))
                .Sum();
        }

        private void PassBackward(Value value)
        {
            _layers
                .OrderByDescending(q => q.LayerIndex)
                .ForEach(q => value = q.PassBackward(value));
        }

        private double[] PassForward(Value value)
        {
            _layers.ForEach(q => value = q.PassForward(value));
            return value.Single;
        }

        private void Learn(double learningRate)
        {
            foreach (var layer in _layers.OfType<ILearnableLayer>())
            {
                layer.UpdateWeights(learningRate);
                layer.UpdateBiases(learningRate);
            }
        }
    }
}
