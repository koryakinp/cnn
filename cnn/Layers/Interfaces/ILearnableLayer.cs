namespace Cnn.Layers.Interfaces
{
    internal interface ILearnableLayer
    {
        void UpdateWeights(double learningRate);
        void UpdateBiases(double learningRate);
    }
}
