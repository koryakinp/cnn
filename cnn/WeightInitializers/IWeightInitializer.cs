namespace Cnn.WeightInitializers
{
    internal interface IWeightInitializer
    {
        double GenerateRandom(double magnitude);
    }
}
