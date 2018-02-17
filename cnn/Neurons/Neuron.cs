using Cnn.Activators;
using System;
using System.Collections.Generic;
using System.Text;

namespace Cnn.Neurons
{
    internal class Neuron
    {
        public double Output { get; private set; }
        public double Bias { get; private set; }
        public double Delta { get; private set; }
        private readonly IActivator _activator;

        public IReadOnlyList<Connection> BackwardConnections { get; set; }
        public IReadOnlyList<Connection> ForwardConnections { get; set; }

        public Neuron(IActivator activator)
        {
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
