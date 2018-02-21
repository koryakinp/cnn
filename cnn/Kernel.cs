using cnn;
using System;

namespace Cnn
{
    internal class Kernel
    {
        public readonly double[,] Weights;
        public readonly double[,] Gradient;

        private double _bias;

        public Kernel(int size)
        {
            _bias = 0;
            Weights = new double[size, size];
            Gradient = new double[size, size];
        }

        public Kernel(double[,] weights, double[,] deltas)
        {
            _bias = 0;
            Weights = weights;
            Gradient = new double[weights.GetLength(0),weights.GetLength(1)];
        }

        public void RandomizeWeights()
        {
            double magnitude = 1 / Math.Sqrt(Weights.GetLength(0) * Weights.GetLength(1));
            Weights.Randomize(magnitude);
        }

        public double GetBias()
        {
            return _bias;
        }
    }
}
