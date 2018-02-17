using System;

namespace Cnn.Activators
{
    public class ActivatorFactory
    {
        public static IActivator Produce(ActivatorType type)
        {
            switch(type)
            {
                case ActivatorType.LogisticActivator: return new LogisticActivator();
                case ActivatorType.TanhActivator: return new TanhActivator();
                case ActivatorType.ReluActivator: return new ReluActivator();
                default: throw new Exception();
            }
        }
    }
}
