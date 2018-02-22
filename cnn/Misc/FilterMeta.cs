namespace Cnn.Misc
{
    internal class FilterMeta
    {
        public readonly int Size;
        public readonly int Channels;

        public FilterMeta(int size, int channels)
        {
            Size = size;
            Channels = channels;
        }
    }
}
