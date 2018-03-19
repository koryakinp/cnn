using Cnn.Layers;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Cnn.Tests
{
    [TestClass]
    public class FlattenLayerTests
    {
        private FlattenLayer _layer { get; set; }
        private readonly MultiValue _multi = new MultiValue(new double[2, 5, 5]
        {
            {
                {-95,-38,47,-65,75},
                {-81,34,-10,-32,-80},
                {-68,-14,-92,11,-64},
                {-59,-74,-36,67,17},
                {95,86,-24,44,94},
            },
            {
                {-62,78,-82,-17,1},
                {14,-78,8,56,31},
                {-93,-76,81,-98,-28},
                {16,60,13,-94,82},
                {41,30,71,2,6},
            }
        });
        private readonly SingleValue _single = new SingleValue(new double[] {
            -95,-38,47,-65,75,
            -81,34,-10,-32,-80,
            -68,-14,-92,11,-64,
            -59,-74,-36,67,17,
            95,86,-24,44,94,
            -62,78,-82,-17,1,
            14,-78,8,56,31,
            -93,-76,81,-98,-28,
            16,60,13,-94,82,
            41,30,71,2,6 });

        [TestInitialize]
        public void SetUp()
        {
            _layer = new FlattenLayer(2, 5, 1);
        }

        [TestMethod]
        public void FlattenLayerForwardPassTest()
        {
            var actual = _layer.PassForward(_multi);
            actual.Single.ForEach((q, i) => Assert.AreEqual(q, _single.Single[i]));
        }

        [TestMethod]
        public void FlattenLayerBackwardPassTest()
        {
            var actual = _layer.PassBackward(_single);
            actual.Multi.ForEach((q, i, j, k) => Assert.AreEqual(_multi.Multi[i, j, k], q));
        }
    }
}
