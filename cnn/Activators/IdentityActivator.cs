namespace Cnn.Activators
{
    internal class IdentityActivator : IActivator
    {
        public double CalculateDeriviative(double input)
        {
            return input;
        }

        public double CalculateValue(double input)
        {
            return 1;
        }
    }
}
