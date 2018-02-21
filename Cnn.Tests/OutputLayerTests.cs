namespace Cnn.Tests
{
    using System;
    using System.Linq;
    using global::Cnn.Activators;
    using global::Cnn.CostFunctions;
    using global::Cnn.Layers;
    using Microsoft.VisualStudio.TestTools.UnitTesting;

    namespace Cnn.Tests
    {
        [TestClass]
        public class OutputLayerTests
        {
            [TestMethod]
            public void OutputLayerForwardPass()
            {
                var layer = new OutputLayer(new QuadraticCostFunction(), 3, 1);
                var actual = layer.PassForward(new SingleValue(new double[] { 1, 2, 3 }));
                var expected = new SingleValue(new double[] { 1, 2, 3 });
                Helper.CompareSingleValues(actual, expected);
            }

            [TestMethod]
            public void OutputLayerBackwardPass()
            {
                var layer = new OutputLayer(new QuadraticCostFunction(), 3, 1);
                layer.SetOutput(new double[] { 6, 5, 4 });
                var actual = layer.PassBackward(new SingleValue(new double[] { 1, 2, 3 }));
               
                var expected = new SingleValue(new double[] { 5,3,1 });
                Helper.CompareSingleValues(actual, expected);
            }

            [TestMethod]
            [ExpectedException(typeof(Exception), Consts.NetworkOutputAndTargetDoNotMatch)]
            public void ShouldThrowExceptionIfOutputsDoNotMatchWithTargets()
            {
                var layer = new OutputLayer(new QuadraticCostFunction(), 3, 1);
                layer.SetOutput(new double[] { 6,5,4,3,2 });
            }
        }
    }
}
