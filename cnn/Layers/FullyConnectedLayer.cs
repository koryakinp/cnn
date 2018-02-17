using cnn;
using Cnn.Activators;
using Cnn.Neurons;
using System.Collections.Generic;
using System.Linq;

namespace Cnn.Layers
{
    internal class FullyConnectedLayer : Layer
    {

        protected readonly IActivator _activator;
        public readonly IReadOnlyList<Neuron> Neurons;

        public FullyConnectedLayer(IActivator activator, int numberOfNeurons, int layerIndex) : base(layerIndex)
        {
            _activator = activator;
            Neurons = Utils.CreateNeurons(numberOfNeurons, activator);
        }

        public override Value PassForward(Value value)
        {
            foreach (var neuron in Neurons)
            {
                double weightedSum = neuron
                    .BackwardConnections
                    .Select((w, j) => value.Single[j] * w.Weight)
                    .Sum();

                neuron.ComputeOutput(weightedSum);
            }

            return new SingleValue(Neurons.Select(q => q.Output).ToArray());
        }

        public override Value PassBackward(Value value)
        {
            foreach (var neuron in Neurons)
            {
                double weightedSum = neuron
                    .ForwardConnections
                    .Select((w, j) => value.Single[j] * w.Weight)
                    .Sum();

                neuron.ComputeDelta(weightedSum);
            }

            return new SingleValue(Neurons.Select(q => q.Delta).ToArray());
        }
    }
}
