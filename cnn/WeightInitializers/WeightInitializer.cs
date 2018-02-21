namespace Cnn.WeightInitializers
{
    internal class WeightInitializer : IWeightInitializer
    {
        public double GenerateRandom(double magnitude)
        {
            return RandomGenerator.Generate(magnitude);
        }
    }
}
