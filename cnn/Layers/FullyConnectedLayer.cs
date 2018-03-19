
using Cnn.Layers.Abstract;
using Cnn.Activators;
using Cnn.Neurons;
using Cnn.WeightInitializers;
using System.Collections.Generic;
using System.Linq;
using Cnn.Layers.Interfaces;
using Cnn.LearningRateAnnealers;

namespace Cnn.Layers
{
    internal class FullyConnectedLayer : Layer, ILearnableLayer
    {
        public readonly IReadOnlyList<Neuron> Neurons;
        private int _numberOfNeuronsInPreviouseLayer;

        public FullyConnectedLayer(
            IActivator activator, 
            int numberOfNeurons, 
            int numberOfNeuronsInPreviouseLayer,
            int layerIndex,
            IWeightInitializer weightInitializer,
            LearningRateAnnealerType lrat) 
            : base(layerIndex)
        {
            _numberOfNeuronsInPreviouseLayer = numberOfNeuronsInPreviouseLayer;

            List<Neuron> neurons = new List<Neuron>();
            for (int i = 0; i < numberOfNeurons; i++)
            {
                neurons.Add(new Neuron(activator, weightInitializer, numberOfNeuronsInPreviouseLayer, lrat));
            }

            Neurons = new List<Neuron>(neurons);
        }

        public override Value PassForward(Value value)
        {
            foreach (var neuron in Neurons)
            {
                double weightedSum = neuron
                    .Weights
                    .Select((w, j) => value.Single[j] * w)
                    .Sum();

                neuron.ComputeOutput(weightedSum);
            }

            return new SingleValue(Neurons.Select(q => q.Output).ToArray());
        }

        public override Value PassBackward(Value value)
        {
            Neurons.ForEach((q, i) => q.ComputeDelta(value.Single[i]));
            double[] deltas = new double[_numberOfNeuronsInPreviouseLayer];
            foreach (var neuron in Neurons)
            {
                neuron
                    .Weights
                    .ForEach((q, i) => deltas[i] += q * neuron.Delta);
            }

            return new SingleValue(deltas);
        }

        public override int GetNumberOfOutputValues() => Neurons.Count;

        public void UpdateWeights(double learningRate)
        {
            Neurons.ForEach(q => q.UpdateWeights());
        }

        public void UpdateBiases(double learningRate)
        {
            Neurons.ForEach(q => q.UpdateBias());
        }
    }
}
