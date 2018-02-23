namespace Cnn.Tests
{
    using global::Cnn.Layers;
    using global::Cnn.Layers.Abstract;
    using global::Cnn.Misc;
    using Microsoft.VisualStudio.TestTools.UnitTesting;

    namespace Cnn.Tests
    {
        [TestClass]
        public class PoolingLayerTests
        {
            private Layer _layer { get; set; }

            [TestInitialize]
            public void SetUp()
            {
                _layer = new PoolingLayer(2, 1, new FilterMeta(4, 2));
            }

            [TestMethod]
            public void PoolingLayerForwardAndBackwardPass()
            {
                var fpActualOutput = _layer.PassForward(new MultiValue(new double[2][,]
                {
                    new double[4,4]
                    {
                        { 5,2,1,3 },
                        { 3,1,2,3 },
                        { 0,6,2,8 },
                        { 3,4,4,2 },
                    },
                    new double[4,4]
                    {
                        { 5,2,1,2 },
                        { 3,1,2,3 },
                        { 0,6,2,8 },
                        { 3,4,3,2 },
                    }
                }));
                var fpExpectedOutput = new MultiValue(new double[2][,]
                {
                    new double[2,2]
                    {
                        { 5,3 },
                        { 6,8 }
                    },
                    new double[2,2]
                    {
                        { 5,3 },
                        { 6,8 }
                    }
                });

                Helper.CompareMultiValues(fpExpectedOutput, fpActualOutput);

                var bpActualOutput = _layer.PassBackward(new MultiValue(new double[2][,]
                {
                    new double[2,2]
                    {
                        { 9,9 },
                        { 9,9 },
                    },
                    new double[2,2]
                    {
                        { 9,9 },
                        { 9,9 }
                    }
                }));
                var bpExpectedOutput = new MultiValue(new double[][,]
{
                    new double[4,4]
                    {
                        { 9,0,0,9 },
                        { 0,0,0,0 },
                        { 0,9,0,9 },
                        { 0,0,0,0 },
                    },
                    new double[4,4]
                    {
                        { 9,0,0,0 },
                        { 0,0,0,9 },
                        { 0,9,0,9 },
                        { 0,0,0,0 },
                    }
});

                Helper.CompareMultiValues(bpExpectedOutput, bpActualOutput);
            }

            [TestMethod]
            public void NumberOfOutputValuesTest()
            {
                Assert.AreEqual(125, new PoolingLayer(3, 1, new FilterMeta(13, 5)).GetNumberOfOutputValues());
                Assert.AreEqual(12, new PoolingLayer(5, 1, new FilterMeta(7, 3)).GetNumberOfOutputValues());
                Assert.AreEqual(200, new PoolingLayer(10, 1, new FilterMeta(100, 2)).GetNumberOfOutputValues());
            }
        }
    }
}
