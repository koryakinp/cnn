using Cnn.Activators;

namespace Cnn.Layers
{
    internal class NonLinearityLayer : Layer
    {
        private readonly IActivator _activator;

        protected NonLinearityLayer(int layerIndex, IActivator activator) : base(layerIndex)
        {
            _activator = activator;
        }

        public override Value PassBackward(Value value)
        {
            foreach (var fm in value.Multi)
            {
                for (int i = 0; i < value.Multi.GetLength(0); i++)
                {
                    for (int j = 0; j < value.Multi.GetLength(1); j++)
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
                for (int i = 0; i < value.Multi.GetLength(0); i++)
                {
                    for (int j = 0; j < value.Multi.GetLength(1); j++)
                    {
                        fm[i, j] = _activator.CalculateValue(fm[i, j]);
                    }
                }
            }

            return value;
        }
    }
}
