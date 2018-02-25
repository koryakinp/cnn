using Cnn.Misc;

namespace Cnn.Layers.Abstract
{
    internal abstract class FilterLayer : Layer
    {
        protected readonly FilterMeta InputFilterMeta;

        protected FilterLayer(int layerIndex, LayerType layerType, FilterMeta filterMeta) 
            : base(layerIndex, layerType)
        {
            InputFilterMeta = filterMeta;
        }

        public abstract FilterMeta GetOutputFilterMeta();

        public override int GetNumberOfOutputValues()
        {
            var fm = GetOutputFilterMeta();
            return fm.Channels * fm.Size * fm.Size;
        }

        protected Value ConvertToMulti(Value value)
        {
            if (!value.IsMulti)
            {
                return value.ToMulti(InputFilterMeta.Size, InputFilterMeta.Channels);
            }
            else
            {
                return value;
            }
        }
    }
}
