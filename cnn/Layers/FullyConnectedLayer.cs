using Cnn.Layers.Abstract;
using Cnn.Activators;
using Cnn.Neurons;
using Cnn.WeightInitializers;
using System.Collections.Generic;
using System.Linq;
using Cnn.Layers.Interfaces;

namespace Cnn.Layers
{
    internal class FullyConnectedLayer : Layer, ILearnableLayer
    {
        public readonly IReadOnlyList<Neuron> Neurons;

        public FullyConnectedLayer(
            IActivator activator, 
            int numberOfNeurons, 
            int numberOfNeuronsInPreviouseLayer,
            int layerIndex,
            IWeightInitializer weightInitializer) 
            : base(layerIndex, LayerType.FullyConnected)
        {
            Neurons = Utils.CreateNeurons(
                numberOfNeurons, 
                numberOfNeuronsInPreviouseLayer, 
                activator,
                weightInitializer);
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

        public override int GetNumberOfOutputValues()
        {
            return Neurons.Count;
        }

        public void UpdateWeights(double learningRate)
        {
            throw new System.NotImplementedException();
        }

        public void UpdateBiases(double learningRate)
        {
            throw new System.NotImplementedException();
        }
    }
}
