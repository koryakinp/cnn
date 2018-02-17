namespace Cnn.Tests
{
    using System;
    using global::Cnn.Misc;
    using Microsoft.VisualStudio.TestTools.UnitTesting;

    namespace Cnn.Tests
    {
        [TestClass]
        public class MatrixProcessorTests
        {
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
                var actual = MatrixProcessor.MaxPool(new double[5, 5]
                {
                    { 5,2,1,9,8 },
                    { 3,1,2,3,5 },
                    { 0,6,2,8,1 },
                    { 3,4,9,2,1 },
                    { 6,3,1,2,2 }
                }, 2);

                var expected = new MaxPoolResult
                {
                    Values = new double[3, 3]
                    {
                        { 5,9,8 },
                        { 6,9,1 },
                        { 6,2,2 }
                    },
                    MaxCoordinates = new Coordinate[]
                    {
                        new Coordinate(0,0),
                        new Coordinate(0,3),
                        new Coordinate(0,4),
                        new Coordinate(2,1),
                        new Coordinate(3,2),
                        new Coordinate(2,4),
                        new Coordinate(4,0),
                        new Coordinate(4,3),
                        new Coordinate(4,4)
                    }
                };

                for (int i = 0; i < actual.Values.GetLength(0); i++)
                {
                    for (int j = 0; j < actual.Values.GetLength(1); j++)
                    {
                        Assert.AreEqual(expected.Values[i, j], actual.Values[i, j]);
                    }
                }

                for (int i = 0; i < actual.MaxCoordinates.Length; i++)
                {
                    Assert.AreEqual(actual.MaxCoordinates[i].X, expected.MaxCoordinates[i].X);
                    Assert.AreEqual(actual.MaxCoordinates[i].Y, expected.MaxCoordinates[i].Y);
                }
            }
        }
    }
}
