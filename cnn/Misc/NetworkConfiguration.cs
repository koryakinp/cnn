using Cnn.CostFunctions;
using Cnn.LearningRateDecayers;

namespace Cnn.Misc
{
    public class NetworkConfiguration
    {
        internal readonly CostFunctionType CostFunctionType;
        internal readonly LearningRateDecayer Decayer;
        internal readonly int InputDimenision;
        internal readonly int InputChannels;

        public NetworkConfiguration(
            CostFunctionType costFunctionType,
            LearningRateDecayer decayer,
            int inputDimenision,
            int inputChannels)
        {
            Decayer = decayer;
            CostFunctionType = costFunctionType;
            InputDimenision = inputDimenision;
            InputChannels = inputChannels;
        }
    }
}
