using System;

namespace Cnn.LearningRateAnnealers
{
    internal class Adagrad : ILearningRateAnnealer
    {
        private double _cache;
        private readonly double _learningRate;

        public Adagrad(double learningRate)
        {
            _learningRate = learningRate;
        }

        public double Compute(double dx)
        {
            _cache += Math.Pow(dx, 2);

            double den = Math.Sqrt(_cache) == 0 
                ? Math.Sqrt(_cache) + double.MinValue 
                : Math.Sqrt(_cache);

            return -_learningRate * dx / den;
        }
    }
}
