using System;

namespace Cnn.LearningRateAnnealers
{
    public class RMSprop : ILearningRateAnnealer
    {
        private double _cache;
        private readonly double _learningRate;
        private readonly double _decayRate;

        public RMSprop(double learningRate, double decayRate)
        {
            _decayRate = decayRate;
            _learningRate = learningRate;
        }

        public double Compute(double dx)
        {
            _cache = _decayRate * _cache + (1 - _decayRate) * Math.Pow(dx, 2);

            double den = Math.Sqrt(_cache) == 0
                ? Math.Sqrt(_cache) + double.MinValue
                : Math.Sqrt(_cache);

            return -_learningRate * dx / den;
        }
    }
}
