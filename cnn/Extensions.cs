using Cnn.Activators;
using System;
using System.Collections.Generic;

namespace Cnn
{
    internal static class Extensions
    {
        public static void AddValue(this double[,] input, double value)
        {
            for (int i = 0; i < input.GetLength(0); i++)
            {
                for (int j = 0; j < input.GetLength(1); j++)
                {
                    input[i, j] += value;
                }
            }
        }

        public static void Activate(this double[,] input, IActivator activator)
        {
            for (int i = 0; i < input.GetLength(0); i++)
            {
                for (int j = 0; j < input.GetLength(1); j++)
                {
                    input[i, j] = activator.CalculateValue(input[i, j]);
                }
            }
        }

        public static void Randomize(this double[,] input, double magnitude)
        {
            for (int i = 0; i < input.GetLength(0); i++)
            {
                for (int j = 0; j < input.GetLength(1); j++)
                {
                    input[i, j] = RandomGenerator.Generate(magnitude);
                }
            }
        }

        public static void ForEach<T>(this IEnumerable<T> source, Action<T> action)
        {
            foreach (T element in source)
                action(element);
        }

        public static void ForEach<T>(this IEnumerable<T> source, Action<T,int> action)
        {
            var i = 0;
            foreach (var e in source) action(e, i++);
        }

        public static List<List<T>> SplitBySize<T>(this List<T> items, int size)
        {
            List<List<T>> res = new List<List<T>>();
            for (int i = 0; i < items.Count; i += size)
            {
                res.Add(items.GetRange(i, Math.Min(size, items.Count - i)));
            }
            return res;
        }

        public static List<List<T>> SplitByStep<T>(this List<T> items, int step)
        {
            List<List<T>> res = new List<List<T>>();
            for (int i = 0; i < step; i++)
            {
                List<T> temp = new List<T>();
                for (int j = i; j < items.Count; j += step)
                {
                    temp.Add(items[j]);
                }
                res.Add(temp);
            }
            return res;
        }

        public static double[,] SingleToDouble(this double[] input, int size)
        {
            var res = new double[size, size];

            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    res[i, j] = input[i * size + j];
                }
            }
            return res;
        }

        public static double[] MultiToSingle(this double[,] input)
        {
            double[] output = new double[input.Length];
            Buffer.BlockCopy(input, 0, output, 0, input.Length);
            return output;
        }


        public static void ForEach(this double[,,] source, Action<double, int, int, int> action)
        {
            for (int d = 0; d < source.GetLength(0); d++)
            {
                for (int w = 0; w < source.GetLength(1); w++)
                {
                    for (int h = 0; h < source.GetLength(2); h++)
                    {
                        action(source[d, w, h], d, w, h);
                    }
                }
            }
        }

        public static void ForEach(this double[,,] source, Action<int, int, int> action)
        {
            for (int d = 0; d < source.GetLength(0); d++)
            {
                for (int w = 0; w < source.GetLength(1); w++)
                {
                    for (int h = 0; h < source.GetLength(2); h++)
                    {
                        action(d, w, h);
                    }
                }
            }
        }
    }
}
