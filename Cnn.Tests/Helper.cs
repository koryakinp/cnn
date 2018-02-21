using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Cnn.Tests
{
    internal static class Helper
    {
        public static void CompareMultiValues(Value expected, Value actual)
        {
            for (int i = 0; i < actual.Multi.Length; i++)
            {
                for (int j = 0; j < actual.Multi[i].GetLength(0); j++)
                {
                    for (int k = 0; k < actual.Multi[i].GetLength(1); k++)
                    {
                        Assert.AreEqual(actual.Multi[i][j, k], expected.Multi[i][j, k]);
                    }
                }
            }
        }

        public static void CompareSingleValues(Value expected, Value actual)
        {
            for (int i = 0; i < actual.Single.Length; i++)
            {
                Assert.AreEqual(actual.Single[i], expected.Single[i]);
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
    }
}
