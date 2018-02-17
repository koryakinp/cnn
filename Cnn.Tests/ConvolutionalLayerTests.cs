namespace Cnn.Tests
{
    using System;
    using global::Cnn.Layers;
    using Microsoft.VisualStudio.TestTools.UnitTesting;

    namespace Cnn.Tests
    {
        [TestClass]
        public class ConvolutionalLayerTests
        {

            [TestMethod]
            public void ForwardAndBackwardPass()
            {
                var layer = new ConvolutionalLayer(2, 2, 1, new double[2][,]
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
                }, null);

                var actualForwardPassOutput = layer.PassForward(new MultiValue(new double[2][,]
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
                }));

                var expectedForwardPassOutput = new MultiValue(new double[][,]
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
                });

                Helper.CompareValues(expectedForwardPassOutput, actualForwardPassOutput);

                var actualBackwardPassOutput = layer.PassBackward(new MultiValue(new double[][,]
                {
                    new double[3,3]
                    {
                        { 2,3,5 },
                        { 2,3,1 },
                        { 4,1,4 }
                    },
                    new double[3,3]
                    {
                        { 4,3,4 },
                        { 2,4,3 },
                        { 2,3,4 }
                    },
                    new double[3,3]
                    {
                        { 2,3,5 },
                        { 2,3,1 },
                        { 4,1,4 }
                    },
                    new double[3,3]
                    {
                        { 4,3,4 },
                        { 2,4,3 },
                        { 2,3,4 }
                    },
                }));

                var expectedBackwardPass = new MultiValue(new double[][,]
                {
                    
                });

                //Helper.CompareValues(expectedBackwardPass, actualBackwardPassOutput);
            }
        }
    }
}
