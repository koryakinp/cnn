using Cnn.Activators;
using Cnn.WeightInitializers;
using System.Collections.Generic;

namespace Cnn.Neurons
{
    internal class Neuron
    {
        public double Output { get; private set; }
        public double Bias { get; private set; }
        public double Delta { get; private set; }
        private readonly IActivator _activator;

        public readonly IReadOnlyList<Connection> BackwardConnections;

        public Neuron(
            IActivator activator, 
            IWeightInitializer weightInitializer, 
            int numberOfConnections)
        {
            BackwardConnections = Utils.CreateConnections(numberOfConnections, weightInitializer);
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
