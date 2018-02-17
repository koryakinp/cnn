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
            public void ForwardAndBackwardPass()
            {
                var expected = new MultiValue(new double[][,]
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

                var input = new MultiValue(new double[2][,]
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
                });

                var gradient = new MultiValue(new double[2][,]
                {
                    new double[4,4]
                    {
                        { 9,9,9,9 },
                        { 9,9,9,9 },
                        { 9,9,9,9 },
                        { 9,9,9,9 },
                    },
                    new double[4,4]
                    {
                        { 9,9,9,9 },
                        { 9,9,9,9 },
                        { 9,9,9,9 },
                        { 9,9,9,9 },
                    }
                });

                var layer = new PoolingLayer(2, 1);

                var fp = layer.PassForward(input);
                var actual = layer.PassBackward(gradient);

                Helper.CompareValues(expected, actual);
            }
        }
    }
}
