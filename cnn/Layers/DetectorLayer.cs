using Cnn.Activators;
using Cnn.Layers.Abstract;
using Cnn.Misc;

namespace Cnn.Layers
{
    internal class DetectorLayer : FilterLayer
    {
        private readonly IActivator _activator;
        private readonly double[][,] _featureMaps;

        public DetectorLayer(int layerIndex, IActivator activator, FilterMeta filterMeta) 
            : base(layerIndex, LayerType.NonLinearity, filterMeta)
        {
            _featureMaps = new double[filterMeta.Channels][,];
            _activator = activator;
        }

        public override FilterMeta GetOutputFilterMeta()
        {
            return InputFilterMeta;
        }

        public override Value PassBackward(Value value)
        {
            value = ConvertToMulti(value);

            for (int k = 0; k < value.Multi.Length; k++)
            {
                for (int i = 0; i < value.Multi[k].GetLength(0); i++)
                {
                    for (int j = 0; j < value.Multi[k].GetLength(1); j++)
                    {
                        value.Multi[k][i, j] = _activator.CalculateDeriviative(_featureMaps[k][i, j]) * value.Multi[k][i, j];
                    }
                }
            }

            return value;
        }

        public override Value PassForward(Value value)
        {
            for (int k = 0; k < value.Multi.Length; k++)
            {
                for (int i = 0; i < value.Multi[k].GetLength(0); i++)
                {
                    for (int j = 0; j < value.Multi[k].GetLength(1); j++)
                    {
                        value.Multi[k][i, j] = _activator.CalculateValue(value.Multi[k][i, j]);
                    }
                }

                _featureMaps[k] = value.Multi[k];
            }

            return value;
        }
    }
}
