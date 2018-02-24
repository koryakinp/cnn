namespace Cnn.Tests
{
    using System.Collections.Generic;
    using System.Linq;
    using global::Cnn.Activators;
    using global::Cnn.Layers;
    using global::Cnn.Layers.Abstract;
    using global::Cnn.WeightInitializers;
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using Moq;

    namespace Cnn.Tests
    {
        [TestClass]
        public class FullyConnectedLayerTests
        {
            private FullyConnectedLayer _layer { get; set; }
            private readonly SingleValue _input = new SingleValue(
                    new double[] { 0.12, 0.29, 0.23, 0.61, 0.11 });

            private readonly SingleValue _delta = new SingleValue(
                    new double[] { 0.35, -0.69, 0.83 });

            [TestInitialize]
            public void SetUp()
            {
                var mock = new Mock<IWeightInitializer>();

                var queue = new Queue<double>(new double[] {
                    0.15, 0.75,-0.33,0.4,0.1,
                    -0.55,-1.09,2.03,0.02,0.42,
                    1.23,-2.93,0.56,-0.98,-0.55
                });

                mock
                    .Setup(q => q.GenerateRandom(It.IsAny<double>()))
                    .Returns(queue.Dequeue);

                _layer = new FullyConnectedLayer(
                    new LogisticActivator(), 3, 5, 1, mock.Object);
            }

            [TestMethod]
            public void FullyConnectedLayerForwardPass()
            {
                var actual = _layer.PassForward(_input);

                var expected = new SingleValue(new double[] { 0.602190358,0.535738948,0.225901511 });

                Helper.CompareSingleValues(actual, expected);
            }

            [TestMethod]
            public void FullyConnectedLayerBackwardPass()
            {
                _layer.PassForward(_input);
                    
                var actual = _layer.PassBackward(_delta);

                var expected = new SingleValue(
                    new double[] { 0.285491826, -0.175318287,-0.294775189, -0.112133648, -0.14352351 });

                Helper.CompareSingleValues(actual, expected);
            }

            [TestMethod]
            public void ShouldUpdateWeights()
            {
                _layer.PassForward(_input);
                _layer.PassBackward(_delta);

                _layer.UpdateWeights(0.1);

                var actual = _layer.PassForward(_input);
                var expected = new SingleValue(new double[3] { 0.603022813, 0.535127639, 0.22279091 });

                Helper.CompareSingleValues(actual, expected);
            }

            [TestMethod]
            public void ShouldUpdateBias()
            {
                _layer.PassForward(_input);
                _layer.PassBackward(_delta);

                _layer.UpdateBiases(0.1);

                var actual = _layer.PassForward(_input);
                var expected = new SingleValue(new double[3] { 0.604197193, 0.531467887, 0.228449704 });

                Helper.CompareSingleValues(actual, expected);
            }

            [TestMethod]
            public void NumberOfOutputValuesTest()
            {
                var actual = _layer.GetNumberOfOutputValues();
                Assert.AreEqual(3, actual);
            }
        }
    }
}
