namespace Cnn.Tests
{
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
        }
    }
}
