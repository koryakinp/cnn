using Cnn.Activators;
using Cnn.Neurons;
using Cnn.WeightInitializers;
using System.Collections.Generic;

namespace Cnn
{
    internal static class Utils
    {
        public static List<Neuron> CreateNeurons(
            int size, 
            int numberOfConnections,
            IActivator activator, 
            IWeightInitializer weightInitializer)
        {
            List<Neuron> output = new List<Neuron>();
            for (int i = 0; i < size; i++)
            {
                output.Add(new Neuron(activator, weightInitializer, numberOfConnections));
            }

            return output;
        }

        public static List<Connection> CreateConnections(int size, IWeightInitializer weightInitializer)
        {
            List<Connection> output = new List<Connection>();
            for (int j = 0; j < size; j++)
            {
                output.Add(new Connection(size, weightInitializer));
            }
            return output;
        }
    }
}