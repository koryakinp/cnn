namespace Cnn.Activators
{
    public interface IActivator
    {
        double CalculateValue(double input);
        double CalculateDeriviative(double input);
    }
}
