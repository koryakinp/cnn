using Cnn.WeightInitializers;
using System;

namespace Cnn
{
    internal class Connection
    {
        public Connection(int numberOfInputs, IWeightInitializer weightInitializer)
        {
            double magnitude = 1 / Math.Sqrt(numberOfInputs);
            Weight = weightInitializer.GenerateRandom(magnitude);
        }

        public double Weight { get; set; }
    }
}
