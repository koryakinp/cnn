namespace Cnn.Layers.Abstract
{
    internal abstract class Layer
    {
        public readonly int LayerIndex;
        public readonly LayerType LayerType;

        protected Layer(int layerIndex, LayerType layerType)
        {
            LayerType = layerType;
            LayerIndex = layerIndex;
        }

        public abstract Value PassForward(Value value);
        public abstract Value PassBackward(Value value);
        public abstract int GetNumberOfOutputValues();
    }
}
