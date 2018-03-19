using Cnn.Activators;
using System;
using System.Collections.Generic;

namespace Cnn
{
    internal static class Extensions
    {
        public static void ForEach<T>(this IEnumerable<T> source, Action<T> action)
        {
            foreach (T element in source)
                action(element);
        }

        public static void ForEach<T>(this IEnumerable<T> source, Action<T, int> action)
        {
            var i = 0;
            foreach (var e in source) action(e, i++);
        }

        public static void ForEach<T>(this T[,,] source, Action<T, int, int, int> action)
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

        public static void ForEach<T>(this T[,,] source, Action<int, int, int> action)
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

        public static void ForEach<T>(this T[,,] source, Action<T> action)
        {
            for (int d = 0; d < source.GetLength(0); d++)
            {
                for (int w = 0; w < source.GetLength(1); w++)
                {
                    for (int h = 0; h < source.GetLength(2); h++)
                    {
                        action(source[d,w,h]);
                    }
                }
            }
        }

        public static void ForEach<T>(this T[,] source, Action<int, int> action)
        {
            for (int w = 0; w < source.GetLength(0); w++)
            {
                for (int h = 0; h < source.GetLength(1); h++)
                {
                    action(w, h);
                }
            }
        }

        public static void ForEach<T>(this T[,] source, Action<T, int, int> action)
        {
            for (int w = 0; w < source.GetLength(0); w++)
            {
                for (int h = 0; h < source.GetLength(1); h++)
                {
                    action(source[w,h],w, h);
                }
            }
        }

        public static T[,] GetSlice<T>(this T[,,] source, int dimension)
        {
            var output = new T[source.GetLength(1), source.GetLength(2)];

            for (int i = 0; i < source.GetLength(1); i++)
            {
                for (int j = 0; j < source.GetLength(2); j++)
                {
                    output[i, j] = source[dimension, i, j];
                }
            }

            return output;
        }
    }
}
