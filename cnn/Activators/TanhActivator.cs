using System;

namespace Cnn.Activators
{
    public class TanhActivator : IActivator
    {
        public double CalculateDeriviative(double input)
        {
            return 1 - Math.Pow(Math.Tanh(input), 2);
        }

        public double CalculateValue(double input)
        {
            return Math.Tanh(input);
        }
    }
}
