using Cnn.LearningRateAnnealers;
using Cnn.WeightInitializers;
using System;

namespace Cnn
{
    internal class Kernel
    {
        public readonly double[,,] Weights;
        public readonly double[,,] Gradient;
        private readonly ILearningRateAnnealer[,,] _learningRateAnnealers;
        private readonly ILearningRateAnnealer _biasLearningRateAnnealer;
        public double BiasGradient;
        public double Bias;

        public Kernel(int size, int depth, LearningRateAnnealerType type)
        {
            _learningRateAnnealers = new ILearningRateAnnealer[depth, size, size];
            _learningRateAnnealers.ForEach((i, j, k) => _learningRateAnnealers[i, j, k] = LearningRateAnnealerFactory.Produce(type));
            Weights = new double[depth, size, size];
            Gradient = new double[depth, size, size];
        }

        public void RandomizeWeights(IWeightInitializer weightInitializer)
        {
            int inputs = Weights.GetLength(0) * Weights.GetLength(1) * Weights.GetLength(2);
            double magnitude = Math.Sqrt((double)1/inputs);
            Weights.ForEach((i, j, k) => Weights[i, j, k] = weightInitializer.GenerateRandom(magnitude));
        }

        public void UpdateWeights()
        {
            Gradient.ForEach((q, i, j, k) => Weights[i, j, k] += _learningRateAnnealers[i, j, k].Compute(q));
        }

        public void UpdateBias()
        {
            double sum = 0;
            Gradient.ForEach(q => sum += q);
            Bias += _biasLearningRateAnnealer.Compute(sum);
        }
    }
}
