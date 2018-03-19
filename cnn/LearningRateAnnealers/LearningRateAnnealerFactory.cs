using System;

namespace Cnn.LearningRateAnnealers
{
    internal static class LearningRateAnnealerFactory
    {
        public static ILearningRateAnnealer Produce(LearningRateAnnealerType type)
        {
            switch(type)
            {
                case LearningRateAnnealerType.Adagrad: return new Adagrad(0.1);
                case LearningRateAnnealerType.RMSprop: return new RMSprop(0.1, 0.99);
                default: throw new Exception("LearningRateAnnealerType is not supported");
            }
        }
    }
}
