namespace Cnn.Tests
{
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using System.Collections.Generic;

    namespace Cnn.Tests
    {
        [TestClass]
        public class MatrixProcessorTests
        {
            [TestMethod]
            public void NewConvolutionTest()
            {
                var input = new double[,,]
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
                    },
                };

                var k1 = new double[,,]
                {
                    {
                        {0,1,1},
                        {-1,0,0},
                        {-1,1,-1},
                    },
                    {
                        { 0,0,-1},
                        { 1,1,1},
                        { 0,-1,-1},
                    },
                };

                var k2 = new double[,,]
                {
                    {
                        {0,-1,1},
                        {-1,-1,-1},
                        {-1,1,0},
                    },
                    {
                        {1,1,1},
                        {-1,-1,0},
                        {-1,1,1},
                    },
                };

                var k3 = new double[,,]
                {
                    {
                        { -1,0,0},
                        { 1,-1,-1},
                        { 1,-1,-1},
                    },
                    {
                        { 0,0,-1},
                        { 1,0,-1},
                        { -1,-1,-1},
                    },
                };

                var e1 = new double[,]
                {
                    {257,-121,407},
                    {-56,-125,2},
                    {-125,-157,-22}
                };

                var e2 = new double[,]
                {
                    {292,-74,-4},
                    {285,-49,287},
                    {-22,-87,-82},
                };

                var e3 = new double[,]
                {
                    {204,157,37},
                    {-101,-85,-72},
                    {-68,124,-310},
                };

                MatrixProcessor.Convolute(input, k1)
                    .ForEach((q, i, j) => Assert.AreEqual(q, e1[i, j]));

                MatrixProcessor.Convolute(input, k2)
                    .ForEach((q, i, j) => Assert.AreEqual(q, e2[i, j]));

                MatrixProcessor.Convolute(input, k3)
                    .ForEach((q, i, j) => Assert.AreEqual(q, e3[i, j]));
            }

            [TestMethod]
            public void NewMaxPoolingTest()
            {
                var input = new double[,,]
                {
                    {
                        { 53, 29, 3,  55, 2 },
                        { 70, 14, 35, 96, 21 },
                        { 54, 64, 66, 70, 16 },
                        { 16, 68, 1,  6,  99 },
                        { 83, 93, 61, 90, 37 }
                    },
                    {
                        { 69, 11, 11, 83, 63 },
                        { 24, 54, 58, 61, 12 },
                        { 96, 7,  1,  36, 2  },
                        { 75, 21, 64, 49, 96 },
                        { 89, 51, 46, 91, 31 },
                    }
                };

                var expected = new double[,,]
                {
                    {
                        { 70, 96, 21 },
                        { 68, 70, 99 },
                        { 93, 90, 37 }
                    },
                    {
                        { 69, 83, 63 },
                        { 96, 64, 96 },
                        { 89, 91, 31 },
                    }
                };

                var expected2 = new bool[,,]
                {
                    {
                        { false, false, false, false, false },
                        { true,  false, false, true,  true },
                        { false, false, false, true,  false },
                        { false, true,  false, false, true },
                        { false, true,  false, true,  true }
                    },
                    {
                        { true,  false, false, true,  true },
                        { false, false, false, false, false },
                        { true,  false, false, false, false  },
                        { false, false, true,  false, true },
                        { true,  false, false, true,  true },
                    }
                };

                var actual = MatrixProcessor.MaxPool(input, 2);

                Helper.CompareArrays(expected, actual.Item1);
                Helper.CompareArrays(expected2, actual.Item2);
            }

            [TestMethod]
            public void NewReverseMaxPoolingTest()
            {
                var input = new double[,,]
                {
                    {
                        { 70, 96, 21 },
                        { 68, 70, 99 },
                        { 93, 90, 37 }
                    },
                    {
                        { 69, 83, 63 },
                        { 96, 64, 96 },
                        { 89, 91, 31 },
                    }
                };

                var expected = new double[,,]
{
                    {
                        { 0,  0,  0,  0, 0 },
                        { 70, 0,  0, 96, 21 },
                        { 0,  0,  0, 70, 0 },
                        { 0,  68, 0,  0, 99 },
                        { 0,  93, 0, 90, 37 }
                    },
                    {
                        { 69, 0, 0, 83, 63 },
                        { 0,  0, 0, 0, 0 },
                        { 96, 0, 0, 0, 0  },
                        { 0,  0, 64, 0, 96 },
                        { 89, 0, 0, 91, 31 },
                    }
                };


                var input2 = new bool[,,]
                {
                    {
                        { false, false, false, false, false },
                        { true, false, false, true, true },
                        { false, false, false, true, false },
                        { false, true, false,  false, true },
                        { false, true, false, true, true }
                    },
                    {
                        { true, false, false, true, true },
                        { false, false, false, false, false },
                        { true, false, false,  false, false  },
                        { false, false, true, false, true },
                        { true, false, false, true, true },
                    }
                };

                var actual = MatrixProcessor.ReverseMaxPool(input, input2, 2);

                Helper.CompareArrays(expected, actual);
            }

            [TestMethod]
            public void ConvolutionTest()
            {
                var input = new double[2][,]
                {
                    new double[5,5]
                    {
                        { 1,1,1,0,0 },
                        { 0,1,1,1,0 },
                        { 0,0,1,1,1 },
                        { 0,0,1,1,0 },
                        { 0,1,1,0,0 }
                    },
                    new double[5,5]
                    {
                        { 1,1,1,0,0 },
                        { 0,1,1,1,0 },
                        { 0,0,1,1,1 },
                        { 0,0,1,1,0 },
                        { 0,1,1,0,0 }
                    },
                };

                var kernels = new double[2][,]
                {
                    new double[3,3]
                    {
                        { 1,0,1 },
                        { 0,1,0 },
                        { 1,0,1 }
                    },
                    new double[3,3]
                    {
                        { 1,0,1 },
                        { 0,1,0 },
                        { 1,0,1 }
                    },
                };

                var expected = new double[][,]
                {
                    new double[3,3]
                    {
                        { 4,3,4 },
                        { 2,4,3 },
                        { 2,3,4 }
                    },
                    new double[3,3]
                    {
                        { 4,3,4 },
                        { 2,4,3 },
                        { 2,3,4 }
                    },
                };

                var actual = MatrixProcessor.Convolute(input, kernels);

                Helper.CompareArrays(expected, actual);
            }

            [TestMethod]
            public void MaxPoolingTest()
            {
                var input = MatrixProcessor.MaxPool(new double[5, 5]
                {
                    { 5,2,1,9,8 },
                    { 3,1,2,3,5 },
                    { 0,6,2,8,1 },
                    { 3,4,9,2,1 },
                    { 6,3,1,2,2 }
                }, 2);

                var actual = MatrixProcessor.ReverseMaxPool(new double[3, 3]
                {
                    { 3,3,3 },
                    { 3,3,3 },
                    { 3,3,3 }
                }, 2, 5, input.MaxCoordinates);

                var expected = new double[5, 5]
                {
                    { 3,0,0,3,3 },
                    { 0,0,0,0,0 },
                    { 0,3,0,0,3 },
                    { 0,0,3,0,0 },
                    { 3,0,0,3,3 }
                };

                for (int i = 0; i < actual.GetLength(0); i++)
                {
                    for (int j = 0; j < actual.GetLength(1); j++)
                    {
                        Assert.AreEqual(actual[i, j], expected[i, j]);
                    }
                }
            }

            [TestMethod]
            public void FlipMatrixTest()
            {
                var initial = new double[2, 3, 3]
                {
                    {
                        { 11,12,13 },
                        { 21,22,23 },
                        { 31,32,33 }
                    },
                    {
                        { 11,12,13 },
                        { 21,22,23 },
                        { 31,32,33 }
                    },
                };
                var expected = new double[2, 3, 3]
{
                    {
                        { 33,32,31 },
                        { 23,22,21 },
                        { 13,12,11 }
                    },
                    {
                        { 33,32,31 },
                        { 23,22,21 },
                        { 13,12,11 }
                    },
                };
                var actual = MatrixProcessor.Flip(initial);
                Helper.CompareArrays(expected, actual);
            }

            [TestMethod]
            public void PadMatrixTest()
            {
                var initial = new double[2, 3, 3]
                {
                    {
                        { 11,12,13 },
                        { 21,22,23 },
                        { 31,32,33 }
                    },
                    {
                        { 11,12,13 },
                        { 21,22,23 },
                        { 31,32,33 }
                    },
                };
                var expected = new double[2, 7, 7]
                {
                    {
                        { 0,0,0 ,0 ,0 ,0,0 },
                        { 0,0,0 ,0 ,0 ,0,0 },
                        { 0,0,11,12,13,0,0 },
                        { 0,0,21,22,23,0,0 },
                        { 0,0,31,32,33,0,0 },
                        { 0,0,0 ,0 ,0 ,0,0 },
                        { 0,0,0 ,0 ,0 ,0,0 },
                    },
                    {
                        { 0,0,0 ,0 ,0 ,0,0 },
                        { 0,0,0 ,0 ,0 ,0,0 },
                        { 0,0,11,12,13,0,0 },
                        { 0,0,21,22,23,0,0 },
                        { 0,0,31,32,33,0,0 },
                        { 0,0,0 ,0 ,0 ,0,0 },
                        { 0,0,0 ,0 ,0 ,0,0 },
                    },
                };
                var actual = MatrixProcessor.Pad(initial, 2);
                Helper.CompareArrays(expected, actual);
            }

            [TestMethod]
            public void UnpadMatrixTest()
            {
                var initial = new double[2, 7, 7]
                {
                    {
                        { 0,0,0 ,0 ,0 ,0,0 },
                        { 0,0,0 ,0 ,0 ,0,0 },
                        { 0,0,11,12,13,0,0 },
                        { 0,0,21,22,23,0,0 },
                        { 0,0,31,32,33,0,0 },
                        { 0,0,0 ,0 ,0 ,0,0 },
                        { 0,0,0 ,0 ,0 ,0,0 },
                    },
                    {
                        { 0,0,0 ,0 ,0 ,0,0 },
                        { 0,0,0 ,0 ,0 ,0,0 },
                        { 0,0,11,12,13,0,0 },
                        { 0,0,21,22,23,0,0 },
                        { 0,0,31,32,33,0,0 },
                        { 0,0,0 ,0 ,0 ,0,0 },
                        { 0,0,0 ,0 ,0 ,0,0 },
                    },
                };

                var expected = new double[2, 3, 3]
                {
                    {
                        { 11,12,13 },
                        { 21,22,23 },
                        { 31,32,33 }
                    },
                    {
                        { 11,12,13 },
                        { 21,22,23 },
                        { 31,32,33 }
                    },
                };

                var actual = MatrixProcessor.Unpad(initial, 2);
                Helper.CompareArrays(expected, actual);
            }
        }
    }
}
