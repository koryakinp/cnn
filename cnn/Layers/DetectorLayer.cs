using Cnn.Activators;
using Cnn.Layers.Abstract;
using Cnn.Misc;

namespace Cnn.Layers
{
    internal class DetectorLayer : FilterLayer
    {
        private readonly IActivator _activator;

        public DetectorLayer(int layerIndex, IActivator activator, FilterMeta filterMeta) 
            : base(layerIndex, LayerType.NonLinearity, filterMeta)
        {
            _activator = activator;
        }

        public override FilterMeta GetOutputFilterMeta()
        {
            return InputFilterMeta;
        }

        public override Value PassBackward(Value value)
        {
            foreach (var fm in value.Multi)
            {
                for (int i = 0; i < fm.GetLength(0); i++)
                {
                    for (int j = 0; j < fm.GetLength(1); j++)
                    {
                        fm[i, j] = _activator.CalculateDeriviative(fm[i, j]);
                    }
                }
            }

            return value;
        }

        public override Value PassForward(Value value)
        {
            foreach (var fm in value.Multi)
            {
                for (int i = 0; i < fm.GetLength(0); i++)
                {
                    for (int j = 0; j < fm.GetLength(1); j++)
                    {
                        fm[i, j] = _activator.CalculateValue(fm[i, j]);
                    }
                }
            }

            return value;
        }
    }
}
