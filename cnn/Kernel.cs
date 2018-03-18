using Cnn.WeightInitializers;
using System;

namespace Cnn
{
    internal class Kernel
    {
        public readonly double[,,] Weights;
        public readonly double[,,] Gradient;
        public double BiasGradient;
        public double Bias;

        public Kernel(int size, int depth)
        {
            Weights = new double[depth, size, size];
            Gradient = new double[depth, size, size];
        }

        public void RandomizeWeights(IWeightInitializer weightInitializer)
        {
            int inputs = Weights.GetLength(0) * Weights.GetLength(1) * Weights.GetLength(2);
            double magnitude = Math.Sqrt((double)1/inputs);
            Weights.ForEach((i, j, k) => Weights[i, j, k] = weightInitializer.GenerateRandom(magnitude));
        }
    }
}
