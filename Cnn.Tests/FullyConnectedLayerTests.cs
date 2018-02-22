namespace Cnn.Tests
{
    using System.Linq;
    using global::Cnn.Activators;
    using global::Cnn.Layers;
    using global::Cnn.WeightInitializers;
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using Moq;

    namespace Cnn.Tests
    {
        [TestClass]
        public class FullyConnectedLayerTests
        {
            [TestMethod]
            public void FullyConnectedLayerForwardPass()
            {
                var mock = new Mock<IWeightInitializer>();

                mock
                    .Setup(q => q.GenerateRandom(It.IsAny<double>()))
                    .Returns(0.10);

                var network = new Network(CostFunctions.CostFunctionType.Quadratic, mock.Object);
                network.AddPoolingLayer(2);
                network.AddFullyConnectedLayer(3, ActivatorType.LogisticActivator);
                network.AddFullyConnectedLayer(3, ActivatorType.LogisticActivator);

                var fc = network._layers.OfType<FullyConnectedLayer>();

                network.TrainModel(new double[2][,]
                {
                    new double[4,4]
                    {
                        { 1,2,3,4 },
                        { 4,3,2,1,},
                        { 1,2,3,4 },
                        { 4,3,2,1 }
                    },
                    new double[4,4]
                    {
                        { 5,6,7,8 },
                        { 8,7,6,5 },
                        { 5,6,7,8 },
                        { 8,7,6,5 }
                    }
                }, new double[2] { 1,0 });
            }
        }
    }
}
