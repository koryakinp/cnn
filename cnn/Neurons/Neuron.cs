using Cnn.Activators;
using Cnn.LearningRateAnnealers;
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
        private readonly ILearningRateAnnealer[] _learningRateAnnealers;
        private readonly ILearningRateAnnealer _biasLearningRateAnnealer;

        public Neuron(
            IActivator activator, 
            IWeightInitializer weightInitializer, 
            int numberOfConnections,
            LearningRateAnnealerType lrat)
        {
            Weights = new double[numberOfConnections];
            _learningRateAnnealers = new ILearningRateAnnealer[numberOfConnections];
            _learningRateAnnealers.ForEach((q, i) => _learningRateAnnealers[i] = LearningRateAnnealerFactory.Produce(lrat));
            _biasLearningRateAnnealer = LearningRateAnnealerFactory.Produce(lrat);

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

        public void UpdateWeights()
        {
            Weights.ForEach((q, i) => Weights[i] += _learningRateAnnealers[i].Compute(Delta));
        }

        public void UpdateBias()
        {
            Bias += _biasLearningRateAnnealer.Compute(Delta);
        }
    }
}
