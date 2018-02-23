using Cnn.WeightInitializers;
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

        public void RandomizeWeights(IWeightInitializer weightInitializer)
        {
            int inputs = Weights.GetLength(0) * Weights.GetLength(1);
            double magnitude = Math.Sqrt((double)1/inputs);

            for (int i = 0; i < Weights.GetLength(0); i++)
            {
                for (int j = 0; j < Weights.GetLength(1); j++)
                {
                    Weights[i, j] = weightInitializer.GenerateRandom(magnitude);
                }
            }
        }

        public double GetBias()
        {
            return _bias;
        }
    }
}
