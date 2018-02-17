using System;

namespace Cnn.Activators
{
    public class ReluActivator : IActivator
    {
        public double CalculateDeriviative(double input)
        {
            return input < 0 ? 0 : 1;
        }

        public double CalculateValue(double input)
        {
            return Math.Max(0, input);
        }
    }
}
