namespace Cnn.Tests
{
    using System;
    using global::Cnn.Layers;
    using Microsoft.VisualStudio.TestTools.UnitTesting;

    namespace Cnn.Tests
    {
        [TestClass]
        public class PoolingLayerTests
        {
            [TestMethod]
            public void PoolingLayerForwardAndBackwardPass()
            {
                var layer = new PoolingLayer(2, 1);

                var fpActualOutput = layer.PassForward(new MultiValue(new double[2][,]
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

                var bpActualOutput = layer.PassBackward(new MultiValue(new double[2][,]
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
        }
    }
}
