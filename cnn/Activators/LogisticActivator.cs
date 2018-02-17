using System;

namespace Cnn.Activators
{
    public class LogisticActivator : IActivator
    {
        public double CalculateDeriviative(double input)
        {
            return input * (1 - input);
        }

        public double CalculateValue(double input)
        {
            return 1 / (1 + Math.Pow(Math.E, -input));
        }
    }
}
