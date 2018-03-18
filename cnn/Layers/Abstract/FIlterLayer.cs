using Cnn.Misc;

namespace Cnn.Layers.Abstract
{
    internal abstract class FilterLayer : Layer
    {
        protected readonly FilterMeta InputFilterMeta;

        protected FilterLayer(int layerIndex, FilterMeta filterMeta) 
            : base(layerIndex)
        {
            InputFilterMeta = filterMeta;
        }

        public abstract FilterMeta GetOutputFilterMeta();

        public override int GetNumberOfOutputValues()
        {
            var fm = GetOutputFilterMeta();
            return fm.Channels * fm.Size * fm.Size;
        }
    }
}
