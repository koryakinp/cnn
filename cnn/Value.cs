using System;

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

        public SingleValue ToSingle()
        {
            if (Multi.Length == 0)
            {
                throw new Exception(Consts.CanNotConvertMultiValueToSingleValue);
            }

            double[] output = new double[Multi.Length * Multi[0].GetLength(0) * Multi[0].GetLength(1)];
            for (int i = 0; i < Multi.Length; i++)
            {
                for (int j = 0; j < Multi[i].GetLength(0); j++)
                {
                    for (int k = 0; k < Multi[i].GetLength(1); k++)
                    {
                        int depth = Multi.Length;
                        int width = Multi[i].GetLength(0);
                        int height = Multi[i].GetLength(1);

                        output[i * width * height + j * width + k] = Multi[i][j, k];
                    }
                }
            }

            return new SingleValue(output);
        }

        public MultiValue ToMulti(int size, int channels)
        {
            var output = new double[channels][,];

            for (int i = 0; i < output.Length; i++)
            {
                output[i] = new double[size, size];

                for (int j = 0; j < size; j++)
                {
                    for (int k = 0; k < size; k++)
                    {
                        output[i][j, k] = Single[i * size * size + j * size + k];
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
        public MultiValue(double[][,] value) : base(value) {}
    }
}
