using cnn;
using Cnn.Activators;
using Cnn.Neurons;
using Cnn.WeightInitializers;
using System.Collections.Generic;
using System.Linq;

namespace Cnn.Layers
{
    internal class FullyConnectedLayer : Layer
    {
        protected readonly IActivator _activator;
        public readonly IReadOnlyList<Neuron> Neurons;
        private readonly IWeightInitializer _weightInitializer;

        public FullyConnectedLayer(
            IActivator activator, 
            int numberOfNeurons, 
            int layerIndex,
            IWeightInitializer weightInitializer) 
            : base(layerIndex, LayerType.FullyConnected)
        {
            _weightInitializer = weightInitializer;
            _activator = activator;
            Neurons = Utils.CreateNeurons(numberOfNeurons, activator);
        }

        public override Value PassForward(Value value)
        {
            if(Neurons.All(q => q.BackwardConnections == null))
            {
                foreach (var neuron in Neurons)
                {
                    neuron.BackwardConnections = Utils
                        .CreateConnections(value.Single.Length, value.Single.Length, _weightInitializer);
                }
            }

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
            Neurons.ForEach((q, i) => q.ComputeDelta(value.Single[i]));
            double[] deltas = new double[Neurons.First().BackwardConnections.Count];
            foreach (var neuron in Neurons)
            {
                neuron
                    .BackwardConnections
                    .ForEach((q, i) => deltas[i] += q.Weight * neuron.Delta);
            }

            return new SingleValue(deltas);
        }
    }
}
