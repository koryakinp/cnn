using System;

namespace Cnn
{
    internal class Connection
    {
        public Connection(int numberOfInputs)
        {
            double magnitutde = 1 / Math.Sqrt(numberOfInputs);
            Weight = RandomGenerator.Generate(magnitutde);
        }

        public double Weight { get; set; }
    }
}
