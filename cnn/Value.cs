using System.Collections.Generic;

namespace Cnn
{
    public abstract class Value
    {
        internal readonly double[] Single;
        internal readonly double[][,] Multi;
        internal readonly bool IsMulti;

        protected Value(double[] value)
        {
            Single = value;
            IsMulti = false;
        }

        protected Value(double[][,] value)
        {
            Multi = value;
            IsMulti = true;
        }
    }

    public class SingleValue : Value
    {
        public SingleValue(double[] value) : base(value) {}
    }

    public class MultiValue : Value
    {
        public MultiValue(double[][,] value) : base(value) {}
    }
}
