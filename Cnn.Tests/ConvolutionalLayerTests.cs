namespace Cnn.Tests
{
    using global::Cnn.Layers;
    using global::Cnn.Layers.Abstract;
    using global::Cnn.Misc;
    using global::Cnn.WeightInitializers;
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using Moq;
    using System;
    using System.Collections.Generic;

    namespace Cnn.Tests
    {
        [TestClass]
        public class ConvolutionalLayerTests
        {
            private ConvolutionalLayer _layer { get; set; }
            private readonly MultiValue _forwardValue = new MultiValue(new double[2,5,5]
            {
                {
                    {-95,-38,47,-65,75},
                    {-81,34,-10,-32,-80},
                    {-68,-14,-92,11,-64},
                    {-59,-74,-36,67,17},
                    {95,86,-24,44,94},
                },
                {
                    {-62,78,-82,-17,1},
                    {14,-78,8,56,31},
                    {-93,-76,81,-98,-28},
                    {16,60,13,-94,82},
                    {41,30,71,2,6},
                }
            });
            private readonly MultiValue _gradients = new MultiValue(new double[3, 3, 3]
            {
                {
                    {-0.74,0.55,0.65},
                    {0.54,0.2,-0.97},
                    {-0.51,0.68,0.01}
                },
                {
                    {-0.27,-0.91,0.91},
                    {0.41,-0.06,0.02},
                    {0.54,0.72,0.18}
                },
                {
                    {0.61,0.78,0.02},
                    {0.13,-0.84,0.66},
                    {0.35,0.13,0.59}
                }
            });

            [TestInitialize]
            public void SetUp()
            {
                var mock = new Mock<IWeightInitializer>();

                var queue = new Queue<double>();

                List<double[,,]> kernels = new List<double[,,]>
                {
                    new double[2,3,3]
                    {
                        {
                            {0,1,1},
                            {-1,0,0},
                            {-1,1,-1}
                        },
                        {
                            {0,0,-1},
                            {1,1,1},
                            {0,-1,-1}
                        }
                    },
                    new double[2,3,3]
                    {
                        {
                            {0,-1,1},
                            {-1,-1,-1},
                            {-1,1,0}
                        },
                        {
                            {1,1,1},
                            {-1,-1,0},
                            {-1,1,1}
                        }
                    },
                    new double[2,3,3]
                    {
                        {
                            {-1,0,0},
                            {1,-1,-1},
                            {1,-1,-1}
                        },
                        {
                            {0,0,-1},
                            {1,0,-1},
                            {-1,-1,-1}
                        }
                    }
                };

                foreach (var kernel in kernels)
                {
                    kernel.ForEach((q) => queue.Enqueue(q));
                }

                mock
                    .Setup(q => q.GenerateRandom(It.IsAny<double>()))
                    .Returns(queue.Dequeue);

                _layer = new ConvolutionalLayer(3, 3, 1, new FilterMeta(5, 2), mock.Object);
            }

            [TestMethod]
            public void ConvolutionalLayerForwarPass()
            {
                var actual = _layer.PassForward(_forwardValue);
                var expected = new MultiValue(new double[3, 3, 3]
                {
                    {
                        {257,-121,407},
                        {-56,-125,2},
                        {-125,-157,-22}
                    },
                    {
                        {292,-74,-4},
                        {285,-49,287},
                        {-22,-87,-82}
                    },
                    {
                        {204,157,37},
                        {-101,-85,-72},
                        {-68,124,-310}
                    }
                });

                actual.Multi.ForEach((q, i, j, k) => Assert.AreEqual(q, expected.Multi[i, j, k]));
            }

            [TestMethod]
            public void ConvolutionalLayerBackwardPass()
            {
                var expected = new double[2, 5, 5]
                {
                    {
                        {-0.61,-1.25,0.43,-0.62,1.56},
                        {1.49,1.77,-1.2,-1.65,-1.88},
                        {0.45,-3.18,-1.18,1.66,-1.16},
                        {-0.5,-2.32,0.58,-2.59,-0.46},
                        {0.32,-1.59,1.83,-1.21,-0.6},
                    },
                    {
                        {-0.27,-1.18,-0.14,-1.33,0.24},
                        {0.55,2.12,-0.43,0.11,0.96},
                        {0.46,0.8,-1.37,-1.86,-1.81},
                        {-1.24,-0.32,-0.84,1.29,-0.25},
                        {-0.89,-0.15,-0.16,-0.51,-0.42},
                    }
                };

                _layer.PassForward(_forwardValue);
                var actual = _layer.PassBackward(_gradients);

                actual.Multi.ForEach((q, i, j, k) => Assert.AreEqual(q, expected[i, j, k], 0.001));
            }

            [TestMethod]
            public void ShouldUpdateWeights()
            {
                throw new NotImplementedException();
            }

            [TestMethod]
            public void ShouldUpdateBiases()
            {
                throw new NotImplementedException();
            }

            [TestMethod]
            public void NumberOfOutputValuesTest()
            {
                var actual = _layer.GetNumberOfOutputValues();
                Assert.AreEqual(27, actual);
            }
        }
    }
}
