using Cnn.Layers.Abstract;

namespace Cnn.Layers
{
    internal class FlattenLayer : Layer
    {
        private readonly int _numberOfchannels;
        private readonly int _channelSize;

        public FlattenLayer(int numberOfchannels, int channelSize, int layerIndex) 
            : base(layerIndex)
        {
            _numberOfchannels = numberOfchannels;
            _channelSize = channelSize;
        }

        public override int GetNumberOfOutputValues()
        {
            return _numberOfchannels * _channelSize * _channelSize;
        }

        public override Value PassBackward(Value value)
        {
            return value.ToMulti(_channelSize, _numberOfchannels);
        }

        public override Value PassForward(Value value)
        {
            return value.ToSingle();
        }
    }
}
