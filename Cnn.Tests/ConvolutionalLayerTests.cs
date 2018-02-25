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
            private ConvolutionalLayer _layer { get; set; }
            private readonly MultiValue _forwardValue = new MultiValue(new double[2][,]
                {
                    new double[4, 4]
                    {
                        { -6, 2,-10,-9 },
                        {  6,-8, 8, 10 },
                        { -2, 3, -4, 5 },
                        {  4, 0, -1, 9 }
                    },
                    new double[4, 4]
                    {
                        { -10, 8, 1, 2 },
                        { -2, 4,  6, 9 },
                        {  5, 10, -7, -5 },
                        {  3, -3, -9, -6 }
                    }
                });
            private readonly MultiValue _gradients = new MultiValue(new double[][,]
                {
                    new double[3,3]
                    {
                        { 0.09, 0.02, 0.04 },
                        { 0, -0.04,-0.08 },
                        { 0.06, 0.01, -0.05 },
                    },
                    new double[3,3]
                    {
                        { 0.06, -0.02, 0.1 },
                        { 0.09,  0.01, 0.05},
                        { -0.06,-0.08, 0},
                    },
                    new double[3,3]
                    {
                        { 0.08,  0.02,  0.03},
                        { 0.07, -0.03,  -0.01},
                        { -0.04,-0.07,  0.04},
                    },
                    new double[3,3]
                    {
                        { -0.06, 0.05, -0.01},
                        { -0.03, 0.10, -0.07},
                        { 0.03,  0.08, -0.02},
                    }
                });

            [TestInitialize]
            public void SetUp()
            {
                var mock = new Mock<IWeightInitializer>();

                var queue = new Queue<double>(new double[] { -2, 0, -5, 3, 7, 5, -4, -6 });

                mock
                    .Setup(q => q.GenerateRandom(It.IsAny<double>()))
                    .Returns(queue.Dequeue);

                _layer = new ConvolutionalLayer(2, 2, 1, new FilterMeta(4, 2), mock.Object);
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
                var actual = _layer.PassBackward(_gradients);
                var expected = new MultiValue(new double[2][,]
                {
                    new double[2,2]
                    {
                        { -2.71, -0.12 },
                        {  1.85,  1.64 },
                    },
                    new double[2,2]
                    {
                        { 1.67, -1.61 },
                        { 1.80, -0.84 },
                    },
                });

                Helper.CompareMultiValues(expected, actual);
            }

            [TestMethod]
            public void ShouldUpdateWeights()
            {
                _layer.PassForward(_forwardValue);
                _layer.PassBackward(_gradients);

                _layer.UpdateWeights(0.1);

                var actual = _layer.PassForward(_forwardValue);
                var expected = new MultiValue(new double[4][,]
                {
                    new double[3,3]
                    {
                        { -54.738,   72.42,     2.1, },
                        {  13.578, -20.079,  29.496, },
                        { -20.784,  -7.866,  43.185, },
                    },
                    new double[3,3]
                    {
                        {  40.398, -14.412,  -5.58 },
                        {   8.211, -89.526, 15.267 },
                        { -35.541, -28.233, 42.579 },
                    },
                    new double[3,3]
                    {
                        { -24.976,  -31.82, -212.165 },
                        {   8.406, -23.968,   98.702 },
                        { -22.633,  13.223,  -56.445 },
                    },
                    new double[3,3]
                    {
                        { -60.674,  17.691, -61.225 },
                        { -78.118,  49.118, 147.289 },
                        {  85.123, 115.949,  -2.702 },
                    }
                });

                Helper.CompareMultiValues(expected, actual);
            }

            [TestMethod]
            public void ShouldUpdateBiases()
            {
                _layer.PassForward(_forwardValue);
                _layer.PassBackward(_gradients);

                _layer.UpdateBiases(0.1);

                var actual = _layer.PassForward(_forwardValue);
                var expected = new MultiValue(new double[4][,]
                {
                    new double[3,3]
                    {
                        {-41.934,60.066,10.066},
                        {7.066,-10.934,19.066},
                        {-15.934,-8.934,40.066}
                    },
                    new double[3,3]
                    {
                        {42.066,-17.934,-4.934},
                        {9.066,-78.934,8.066},
                        {-33.934,-31.934,41.066}
                    },
                    new double[3,3]
                    {
                        {-7.898,-51.898,-206.898},
                        {-7.898,-3.898,92.102},
                        {-14.898,7.102,-52.898}
                    },
                    new double[3,3]
                    {
                        {-45.898,9.102,-60.898},
                        {-73.898,60.102,145.102},
                        {91.102,101.102,-1.898}
                    }
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
