namespace Cnn.Tests
{
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
            private Layer _layer { get; set; }

            [TestInitialize]
            public void SetUp()
            {
                var mock = new Mock<IWeightInitializer>();

                mock
                    .Setup(q => q.GenerateRandom(It.IsAny<double>()))
                    .Returns(0.10);

                _layer = new FullyConnectedLayer(
                    new LogisticActivator(), 5, 10, 1, mock.Object);
            }

            [TestMethod]
            public void FullyConnectedLayerForwardPass()
            {
                var actual = _layer.PassForward(new SingleValue(
                    new double[] { 0.12, 0.29, 0.23, 0.61, 0.11, 0.18, 0.67, 0.23, 0.87, 0.31 }));

                var expected = new SingleValue(
                    new double[] { 0.589524491, 0.589524491, 0.589524491, 0.589524491, 0.589524491 });

                Helper.CompareSingleValues(actual, expected);
            }

            [TestMethod]
            public void FullyConnectedLayerBackwardPass()
            {
                _layer.PassForward(new SingleValue(
                    new double[] { 0.12, 0.29, 0.23, 0.61, 0.11, 0.18, 0.67, 0.23, 0.87, 0.31 }));

                var actual = _layer.PassBackward(new SingleValue(
                    new double[] { 0.51, 0.30, 0.14, 0.18, 0.29 }));

                var expected = new SingleValue(
                    new double[] {
                        0.034361922, 0.034361922, 0.034361922, 0.034361922, 0.034361922,
                        0.034361922, 0.034361922, 0.034361922, 0.034361922, 0.034361922
                    });

                Helper.CompareSingleValues(actual, expected);
            }

            [TestMethod]
            public void NumberOfOutputValuesTest()
            {
                var actual = _layer.GetNumberOfOutputValues();
                Assert.AreEqual(5, actual);
            }
        }
    }
}
