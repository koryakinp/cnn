using Cnn.Activators;
using Cnn.CostFunctions;
using Cnn.LearningRateAnnealers;
using Cnn.LearningRateDecayers;
using Cnn.Misc;

namespace Cnn.Example
{
    class Program
    {
        static void Main(string[] args)
        {
            var config = new NetworkConfiguration(
                CostFunctionType.Quadratic,
                new FlatDecayer(0.5),
                28, 1);

            var network = new Network(config);

            network.AddConvolutionalLayer(5,5, LearningRateAnnealerType.Adagrad);
            network.AddPoolingLayer(2);
            network.AddDetectorLayer(ActivatorType.LogisticActivator);
            network.AddConvolutionalLayer(5,3, LearningRateAnnealerType.Adagrad);
            network.AddPoolingLayer(2);
            network.AddDetectorLayer(ActivatorType.LogisticActivator);
            network.AddFullyConnectedLayer(5, ActivatorType.LogisticActivator, LearningRateAnnealerType.Adagrad);
            network.AddFullyConnectedLayer(4, ActivatorType.LogisticActivator, LearningRateAnnealerType.Adagrad);

            foreach (var item in MnistReader.ReadTrainingData())
            {
                var data = item.Data.ConvertPixels();
                var input = new double[1, data.GetLength(0), data.GetLength(1)];

                for (int i = 0; i < data.GetLength(0); i++)
                {
                    for (int j = 0; j < data.GetLength(1); j++)
                    {
                        input[0, i, j] = data[i, j];
                    }
                }

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
