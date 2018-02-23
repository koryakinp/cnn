namespace Cnn.Tests
{
    using global::Cnn.Layers;
    using global::Cnn.Layers.Abstract;
    using global::Cnn.Misc;
    using global::Cnn.WeightInitializers;
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using Moq;
    using System.Collections.Generic;

    namespace Cnn.Tests
    {
        [TestClass]
        public class ConvolutionalLayerTests
        {

            private Layer _layer { get; set; }
            private MultiValue _forwardValue { get; set; }

            [TestInitialize]
            public void SetUp()
            {
                var mock = new Mock<IWeightInitializer>();

                var queue = new Queue<double>(new double[] { -2, 0, -5, 3, 7, 5, -4, -6 });

                mock
                    .Setup(q => q.GenerateRandom(It.IsAny<double>()))
                    .Returns(queue.Dequeue);

                _layer = new ConvolutionalLayer(2, 2, 1, new FilterMeta(4, 2), mock.Object);

                _forwardValue = new MultiValue(new double[2][,]
                {
                    new double[4,4]
                    {
                        { -6, 2,-10,-9 },
                        {  6,-8, 8, 10 },
                        { -2, 3, -4, 5 },
                        {  4, 0, -1, 9 }
                    },
                    new double[4,4]
                    {
                        { -10, 8, 1, 2 },
                        { -2, 4,  6, 9 },
                        {  5, 10, -7, -5 },
                        {  3, -3, -9, -6 }
                    }
                });
            }

            [TestMethod]
            public void ConvolutionalLayerForwarPass()
            {
                var actual = _layer.PassForward(_forwardValue);

                var expected = new MultiValue(new double[][,]
                {
                    new double[3,3]
                    {
                        { -42, 60,  10 },
                        {  7,  -11, 19 },
                        { -16, -9,  40 },

                    },
                    new double[3,3]
                    {
                        {  42,  -18, -5 },
                        {  9,   -79, 8 },
                        {  -34, -32, 41 }
                    },
                    new double[3,3]
                    {
                        { -8,  -52, -207 },
                        { -8,  -4,  92 },
                        { -15, 7,   -53 },
                    },
                    new double[3,3]
                    {
                        { -46, 9, -61 },
                        { -74, 60,  145 },
                        {  91, 101, -2 },
                    },
                });

                Helper.CompareMultiValues(expected, actual);
            }

            [TestMethod]
            public void ConvolutionalLayerBackwardPass()
            {
                _layer.PassForward(_forwardValue);
                var actual = _layer.PassBackward(new MultiValue(new double[][,]
                {
                    new double[3,3]
                    {
                        { 9, 2, 4 },
                        { 0, -4,-8 },
                        { 6, 1, -5 },
                    },
                    new double[3,3]
                    {
                        { 6, -2, 10 },
                        { 9,  1, 5},
                        { -6,-8, 0},
                    },
                    new double[3,3]
                    {
                        { 8,  2,  3},
                        { 7, -3,  -1},
                        { -4,-7,  4},
                    },
                    new double[3,3]
                    {
                        { -6, 5, -1},
                        { -3, 10, -7},
                        { 3,  8, -2},
                    }
                }));
                var expected = new MultiValue(new double[2][,]
                {
                    new double[2,2]
                    {
                        { -271,  -12 },
                        { 185, 164 },
                    },
                    new double[2,2]
                    {
                        { 167, -161 },
                        { 180, -84 },
                    },
                });

                Helper.CompareMultiValues(expected, actual);
            }

            [TestMethod]
            public void NumberOfOutputValuesTest()
            {
                var actual = _layer.GetNumberOfOutputValues();
                Assert.AreEqual(36, actual);
            }
        }
    }
}
