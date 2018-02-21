using System;

namespace Cnn.CostFunctions
{
    internal class QuadraticCostFunction : ICostFunction
    {
        public double ComputeDeriviative(double target, double output)
        {
            return output - target;
        }

        public double ComputeValue(double target, double output)
        {
            return 0.5 * Math.Pow((output - target), 2);
        }
    }
}
