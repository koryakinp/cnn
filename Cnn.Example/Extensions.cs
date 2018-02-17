using System;
using System.IO;

namespace Cnn.Example
{
    public static class Extensions
    {
        public static int ReadBigInt32(this BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(Int32));
            if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }

        public static double[,] ConvertPixels(this byte[,] input)
        {
            double[,] output = new double[input.GetLength(0), input.GetLength(1)];

            for (int i = 0; i < input.GetLongLength(0); i++)
            {
                for (int j = 0; j < input.GetLongLength(1); j++)
                {
                    output[i, j] = (double)input[i, j] / 255;
                }
            }

            return output;
        }
    }
}
