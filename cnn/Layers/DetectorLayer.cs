using Cnn.Activators;
using Cnn.Layers.Abstract;
using Cnn.Misc;

namespace Cnn.Layers
{
    internal class DetectorLayer : FilterLayer
    {
        private readonly IActivator _activator;
        private readonly double[,,] _featureMaps;

        public DetectorLayer(int layerIndex, IActivator activator, FilterMeta filterMeta) 
            : base(layerIndex, filterMeta)
        {
            _featureMaps = new double[filterMeta.Channels,filterMeta.Size, filterMeta.Size];
            _activator = activator;
        }

        public override FilterMeta GetOutputFilterMeta()
        {
            return InputFilterMeta;
        }

        public override Value PassBackward(Value value)
        {
            value.Multi.ForEach((q, k, i, j) =>
                value.Multi[k, i, j] = _activator.CalculateDeriviative(_featureMaps[k, i, j]) * value.Multi[k, i, j]);

            return value;
        }

        public override Value PassForward(Value value)
        {
            _featureMaps.ForEach((k,i,j) =>
                _featureMaps[k,i,j] = _activator.CalculateValue(value.Multi[k,i,j]));

            return new MultiValue(_featureMaps);
        }
    }
}
