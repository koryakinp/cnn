using System;

namespace Cnn
{
    public abstract class Value
    {
        internal readonly double[] Single;
        internal readonly double[,,] Multi;
        internal readonly bool IsMulti;

        protected Value(double[] value)
        {
            Single = value;
            IsMulti = false;
        }

        protected Value(double[,,] value)
        {
            Multi = value;
            IsMulti = true;
        }

        public SingleValue ToSingle()
        {
            if (Multi.Length == 0)
            {
                throw new Exception(Consts.CanNotConvertMultiValueToSingleValue);
            }

            double[] output = new double[Multi.Length * Multi.GetLength(1) * Multi.GetLength(1)];

            Multi.ForEach((i, j, k) =>
            {
                int depth = Multi.GetLength(0);
                int width = Multi.GetLength(1);
                int height = Multi.GetLength(2);

                output[i * width * height + j * width + k] = Multi[i, j, k];
            });

            return new SingleValue(output);
        }

        public MultiValue ToMulti(int size, int channels)
        {
            var output = new double[channels,size,size];

            for (int i = 0; i < output.Length; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    for (int k = 0; k < size; k++)
                    {
                        output[i,j,k] = Single[i * size * size + j * size + k];
                    }
                }
            }

            return new MultiValue(output);
        }
    }

    public class SingleValue : Value
    {
        public SingleValue(double[] value) : base(value) {}
    }

    public class MultiValue : Value
    {
        public MultiValue(double[,,] value) : base(value) {}
    }
}
