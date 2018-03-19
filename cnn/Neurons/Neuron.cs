using Cnn.Activators;
using Cnn.WeightInitializers;
using System;

namespace Cnn.Neurons
{
    internal class Neuron
    {
        public double Output { get; private set; }
        public double Bias { get; set; }
        public double Delta { get; private set; }
        public readonly double[] Weights;
        private readonly IActivator _activator;

        public Neuron(IActivator activator, IWeightInitializer weightInitializer, int numberOfConnections)
        {
            Weights = new double[numberOfConnections];
            double magnitude = 1 / Math.Sqrt(numberOfConnections);
            Weights.ForEach((q, i) => Weights[i] = weightInitializer.GenerateRandom(magnitude));
            _activator = activator;
        }

        public void ComputeOutput(double weightedSum)
        {
            Output = _activator.CalculateValue(weightedSum + Bias);
        }

        public void ComputeDelta(double weightedDelta)
        {
            Delta = weightedDelta * _activator.CalculateDeriviative(Output);
        }
    }
}
