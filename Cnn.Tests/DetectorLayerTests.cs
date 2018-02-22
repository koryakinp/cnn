namespace Cnn.Tests
{
    using global::Cnn.Activators;
    using global::Cnn.Layers;
    using global::Cnn.Misc;
    using Microsoft.VisualStudio.TestTools.UnitTesting;

    namespace Cnn.Tests
    {
        [TestClass]
        public class DetectorLayerTests
        {
            [TestMethod]
            public void DetectorLayerForwardPass()
            {
                var layer = new DetectorLayer(1, new LogisticActivator(), new FilterMeta(3,2));
                var actual = layer.PassForward(new MultiValue(new double[2][,]
                {
                    new double[3,3]
                    {
                        { 4, 5, 7 },
                        { -1, 4, -3 },
                        { 2, 0, -6 }
                    },
                    new double[3,3]
                    {
                        { -10, 8, -1, },
                        {   8, -9,-7 },
                        {  8,  7, -6 },
                    }
                }));

                var expected = new MultiValue(new double[2][,]
                {
                    new double[3,3]
                    {
                        { 0.982013, 0.993307, 0.999088 },
                        { 0.268941, 0.982013, 0.047425 },
                        { 0.880797, 0.500000, 0.002472 }
                    },
                    new double[3,3]
                    {
                        { 0.000045, 0.999664, 0.268941 },
                        { 0.999664, 0.000123, 0.000911 },
                        { 0.999664, 0.999088, 0.002472 }
                    },
                });

                Helper.CompareMultiValues(expected, actual);
            }

            [TestMethod]
            public void DetectorLayerBackwardPass()
            {

            }
        }
    }
}
