namespace Cnn.Tests
{
    using global::Cnn.Activators;
    using global::Cnn.Layers;
    using global::Cnn.Layers.Abstract;
    using global::Cnn.Misc;
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using Moq;

    namespace Cnn.Tests
    {
        [TestClass]
        public class DetectorLayerTests
        {
            private Layer _layer { get; set; }

            private readonly MultiValue input = new MultiValue(new double[,,]
            {
                {
                    { 4, 5, 7 },
                    { -1, 4, -3 },
                    { 2, 0, -6 }
                },
                {
                    { -10, 8, -1, },
                    {   8, -9,-7 },
                    {  8,  7, -6 },
                }
            });

            private readonly MultiValue delta = new MultiValue(new double[,,]
            {
                {
                    { -2, 2, 0 },
                    { 5, -5, 1 },
                    { -8, -7, 8 }
                },
                {
                    { 7, 3, 9, },
                    { 6, -3,-9 },
                    { 4, -4, -6 },
                }
            });

            [TestInitialize]
            public void SetUp()
            {
                _layer = new DetectorLayer(1, new LogisticActivator(), new FilterMeta(3, 2));
            }

            [TestMethod]
            public void DetectorLayerForwardPass()
            {
                var actual = _layer.PassForward(input);

                var expected = new MultiValue(new double[,,]
                {
                    {
                        { 0.982013, 0.993307, 0.999088 },
                        { 0.268941, 0.982013, 0.047425 },
                        { 0.880797, 0.500000, 0.002472 }
                    },
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
                _layer.PassForward(input);
                var actual = _layer.PassBackward(delta);

                var expected = new MultiValue(new double[,,]
                {
                    {
                        { -0.035325412,  0.013296113, 0          },
                        {  0.983059666, -0.088313531, 0.04517666 },
                        { -0.839948683, -1.75,        0.01973207 },
                    },
                    {
                        { 0.000317771,  0.001005713, 1.769507399  },
                        { 0.002011426, -0.000370138, -0.008191991 },
                        { 0.001340951, -0.003640885, -0.014799056 },
                    },
                });

                Helper.CompareMultiValues(expected, actual);
            }

            [TestMethod]
            public void NumberOfOutputValuesTest()
            {
                var actual = _layer.GetNumberOfOutputValues();
                Assert.AreEqual(18, actual);
            }
        }
    }
}
