using Cnn.WeightInitializers;
using System;

namespace Cnn
{
    internal class Kernel
    {
        public readonly double[,] Weights;
        public readonly double[,] Gradient;
        public double BiasGradient;
        public double Bias;

        public Kernel(int size)
        {
            Weights = new double[size, size];
            Gradient = new double[size, size];
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
    }
}
