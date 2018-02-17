using Cnn.Activators;
using Cnn.Neurons;
using System.Collections.Generic;

namespace Cnn
{
    internal static class Utils
    {
        public static List<Neuron> CreateNeurons(int size, IActivator activator)
        {
            List<Neuron> output = new List<Neuron>();
            for (int i = 0; i < size; i++)
            {
                output.Add(new Neuron(activator));
            }

            return output;
        }

        public static List<Connection> CreateConnections(int size, int numberOfInputs)
        {
            List<Connection> output = new List<Connection>();
            for (int j = 0; j < size; j++)
            {
                output.Add(new Connection(numberOfInputs));
            }
            return output;
        }
    }
}