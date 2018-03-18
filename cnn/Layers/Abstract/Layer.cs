namespace Cnn.Layers.Abstract
{
    internal abstract class Layer
    {
        public readonly int LayerIndex;

        protected Layer(int layerIndex)
        {
            LayerIndex = layerIndex;
        }

        public abstract Value PassForward(Value value);
        public abstract Value PassBackward(Value value);
        public abstract int GetNumberOfOutputValues();
    }
}
