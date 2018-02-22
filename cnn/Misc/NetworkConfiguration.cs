using Cnn.CostFunctions;

namespace Cnn.Misc
{
    public class NetworkConfiguration
    {
        public readonly CostFunctionType CostFunctionType;
        public readonly int InputDimenision;
        public readonly int InputChannels;

        public NetworkConfiguration(
            CostFunctionType costFunctionType,
            int inputDimenision,
            int inputChannels
            )
        {
            CostFunctionType = costFunctionType;
            InputDimenision = inputDimenision;
            InputChannels = inputChannels;
        }
    }
}
