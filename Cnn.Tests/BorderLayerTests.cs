namespace Cnn.Tests
{
    using System;
    using global::Cnn.Layers;
    using Microsoft.VisualStudio.TestTools.UnitTesting;

    namespace Cnn.Tests
    {
        [TestClass]
        public class BorderLayerTests
        {
            [TestMethod]
            public void ShouldConvertMultiToSingle()
            {
                BorderLayer bl = new BorderLayer(1);
                MultiValue multi = new MultiValue(new double[2][,]
                {
                    new double[3,3]
                    {
                        { 1,4,2 },
                        { 3,5,2 },
                        { 3,4,9 },
                    },
                    new double[3,3]
                    {
                        { 5,2,1 },
                        { 3,1,2 },
                        { 0,6,2 },
                    }
                });
                SingleValue expected = new SingleValue(new double[] {1,4,2,3,5,2,3,4,9,5,2,1,3,1,2,0,6,2});

                var actual = bl.PassForward(multi);

                for (int i = 0; i < expected.Single.Length; i++)
                {
                    Assert.AreEqual(actual.Single[i], expected.Single[i]);
                }
            }

            [TestMethod]
            public void ShouldConvertSingleToMulti()
            {
                BorderLayer bl = new BorderLayer(1);
                SingleValue input = new SingleValue(new double[] {1,4,2,3,5,2,3,4,9,5,2,1,3,1,2,0,6,2});
                MultiValue expected = new MultiValue(new double[2][,]
                {
                    new double[3,3]
                    {
                        { 1,4,2 },
                        { 3,5,2 },
                        { 3,4,9 },
                    },
                    new double[3,3]
                    {
                        { 5,2,1 },
                        { 3,1,2 },
                        { 0,6,2 },
                    }
                });

                bl.PassForward(expected);
                var actual = bl.PassBackward(input);

                Helper.CompareValues(expected, actual);
            }

            [TestMethod]
            public void ForwardAndBackwardPass()
            {
                var input = new MultiValue(new double[2][,]
                {
                    new double[3,3]
                    {
                        { 1,2,3 },
                        { 1,2,3 },
                        { 1,2,3 }
                    },
                    new double[3,3]
                    {
                        { 1,2,3 },
                        { 1,2,3 },
                        { 1,2,3 }
                    }
                });

                BorderLayer bp = new BorderLayer(1);

                var single = bp.PassForward(input);
                var output = bp.PassBackward(single);

                Helper.CompareValues(input, output);
            }
        }
    }
}
