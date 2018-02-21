using Cnn.Activators;
using Cnn.CostFunctions;
using System.Collections.Generic;
using System.Linq;

namespace Cnn.Example
{
    class Program
    {
        static void Main(string[] args)
        {
            Network network = new Network(CostFunctionType.Quadratic);
            network.AddConvolutionalLayer(5,5);
            network.AddPoolingLayer(2);
            network.AddConvolutionalLayer(5,3);
            network.AddPoolingLayer(2);
            network.AddFullyConnectedLayer(5, ActivatorType.LogisticActivator);
            network.AddFullyConnectedLayer(4, ActivatorType.LogisticActivator);
            network.AddOutputLayer(10, ActivatorType.LogisticActivator);

            foreach (var item in MnistReader.ReadTrainingData())
            {
                var input = new double[][,] { item.Data.ConvertPixels() };
                var error = network.TrainModel(input, GetTargetOutput(item.Label));
            }
        }

        private static double[] GetTargetOutput(int label)
        {
            var output = new double[10];

            for (int i = 0; i <= 9; i++)
            {
                output[i] = i == label ? 1 : 0;
            }

            return output;
        }
    }
}
