using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Cnn.Tests
{
    internal static class Helper
    {
        public static void CompareMultiValues(Value expected, Value actual, double delta = 0.000001)
        {
            actual.Multi.ForEach((i,j,k) => Assert.AreEqual(actual.Multi[i,j,k], expected.Multi[i,j,k], delta));
        }

        public static void CompareSingleValues(Value expected, Value actual, double delta = 0.000001)
        {
            for (int i = 0; i < actual.Single.Length; i++)
            {
                Assert.AreEqual(actual.Single[i], expected.Single[i], delta);
            }
        }

        public static void CompareArrays(double[][,] expected, double[][,] actual)
        {
            for (int k = 0; k < actual.Length; k++)
            {
                for (int i = 0; i < actual[k].GetLength(0); i++)
                {
                    for (int j = 0; j < actual[k].GetLength(1); j++)
                    {
                        Assert.AreEqual(actual[k][i, j], expected[k][i, j]);
                    }
                }
            }
        }

        public static void CompareArrays(double[,,] expected, double[,,] actual)
        {
            Assert.AreEqual(expected.Length, actual.Length);

            for (int k = 0; k < expected.GetLength(0); k++)
            {
                for (int i = 0; i < expected.GetLength(1); i++)
                {
                    for (int j = 0; j < expected.GetLength(2); j++)
                    {
                        Assert.AreEqual(actual[k,i,j], expected[k,i,j]);
                    }
                }
            }
        }

        public static void CompareArrays(bool[,,] expected, bool[,,] actual)
        {
            for (int k = 0; k < actual.GetLength(0); k++)
            {
                for (int i = 0; i < actual.GetLength(1); i++)
                {
                    for (int j = 0; j < actual.GetLength(2); j++)
                    {
                        Assert.AreEqual(actual[k, i, j], expected[k, i, j]);
                    }
                }
            }
        }
    }
}
