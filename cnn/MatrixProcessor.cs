using Cnn.Misc;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Cnn
{
    internal static class MatrixProcessor
    {
        public static (double[,,], bool[,,]) MaxPool(double[,,] input, int kernelSize)
        {
            int size = input.GetLength(1) % kernelSize == 0
                ? input.GetLength(1) / kernelSize
                : (input.GetLength(1) / kernelSize) + 1;

            var max = new bool[input.GetLength(0), input.GetLength(1), input.GetLength(2)];
            var output = new double[input.GetLength(0), size, size];

            for (int i = 0; i < input.GetLength(0); i++)
            {
                for (int j = 0, jj = 0; j < input.GetLength(1); j += kernelSize, jj++)
                {
                    for (int k = 0, kk = 0; k < input.GetLength(2); k += kernelSize, kk++)
                    {
                        var res = GetMax(i, j, k, kernelSize, input);
                        output[i, jj, kk] = res.max;
                        max[i, res.x, res.y] = true;
                    }
                }
            }

            return (output, max);
        }

        public static double[,] Convolute(double[,,] input, double[,,] kernels)
        {
            int outputSize = input.GetLength(1) - kernels.GetLength(2) + 1;
            int kernelSize = kernels.GetLength(2);
            int inputSize = input.GetLength(1);

            var output = new double[outputSize, outputSize];

            for (int i = 0; i < kernels.GetLength(0); i++)
            {
                for (int j = 0; j <= inputSize - kernelSize; j++)
                {
                    for (int k = 0; k <= inputSize - kernelSize; k++)
                    {
                        for (int jj = 0; jj < kernelSize; jj++)
                        {
                            for (int kk = 0; kk < kernelSize; kk++)
                            {
                                output[j, k] += input[i, j + jj, k + kk] * kernels[i, jj, kk];
                            }
                        }
                    }
                }
            }

            return output;
        }

        private static double Convolute(int i, int j, int k, double[,,] input, double[,,] kernels, int kernel)
        {
            int size = kernels.GetLength(1);
            double output = 0;

            for (int jj = 0; j < j + size; j++, jj++)
            {
                for (int kk = 0; k < k + size; k++, kk++)
                {
                    output += input[i, j, k] * kernels[kernel, jj, kk];
                }
            }

            return output;
        }

        private static (double max, int x, int y) GetMax(int i, int j, int k, int size, double[,,] arr)
        {
            double max = double.MinValue;
            int x = 0;
            int y = 0;

            for (int jj = j; jj < j + size && jj < arr.GetLength(1); jj++)
            {
                for (int kk = k; kk < k + size && kk < arr.GetLength(2); kk++)
                {
                    if (arr[i, jj, kk] > max)
                    {
                        max = arr[i, jj, kk];
                        x = jj;
                        y = kk;
                    }
                }
            }

            return (max, x, y);
        }

        public static List<MaxPoolResult> MaxPool(double[][,] input, int kernelSize)
        {
            int height = input[0].GetLength(0);
            int width = input[0].GetLength(1);

            if (height < kernelSize || width < kernelSize)
            {
                throw new InvalidOperationException(Consts.FeatureMapMustBeBiggerThanKernel);
            }

            return input
                .Select(q => MaxPool(q, kernelSize))
                .ToList();
        }

        public static MaxPoolResult MaxPool(double[,] input, int kernelSize)
        {
            int width = input.GetLength(0);
            int height = input.GetLength(1);

            int outputWidth = width % kernelSize == 0 ? width / kernelSize : width / kernelSize + 1;
            int outputHeight = height % kernelSize == 0 ? height / kernelSize : height / kernelSize + 1;

            MaxPoolResult res = new MaxPoolResult()
            {
                Values = new double[outputWidth, outputHeight],
                MaxCoordinates = new Coordinate[outputWidth * outputHeight]
            };

            for (int i = 0, oi = 0; i < height; i += kernelSize, oi++)
            {
                for (int j = 0, oj = 0; j < width; j += kernelSize, oj++)
                {
                    double curMax = double.MinValue;
                    for (int k = 0; k < kernelSize; k++)
                    {
                        for (int z = 0; z < kernelSize; z++)
                        {
                            int y = i + k;
                            int x = j + z;

                            if (y < height && x < width && input[y, x] > curMax)
                            {
                                curMax = input[y, x];
                                res.MaxCoordinates[oi * outputWidth + oj] = new Coordinate(y, x);
                                res.Values[oi, oj] = input[y, x];
                            }
                        }
                    }
                }
            }

            return res;
        }

        public static double[,,] ReverseMaxPool(double[,,] input, bool[,,] max, int kernelSize)
        {
            var output = new double[max.GetLength(0), max.GetLength(1), max.GetLength(2)];

            output.ForEach((k, i, j) =>
            {
                if (max[k, i, j])
                {
                    int ii = i / kernelSize;
                    int jj = j / kernelSize;

                    output[k, i, j] = input[k, ii, jj];
                }
            });

            return output;
        }

        public static double[,] ReverseMaxPool(double[,] input, int kernelSize, int originalSize, Coordinate[] maxCoordinates)
        {
            var output = new double[originalSize, originalSize];

            for (int i = 0; i < input.GetLength(0); i++)
            {
                for (int j = 0; j < input.GetLength(1); j++)
                {
                    for (int ki = 0; ki < kernelSize; ki++)
                    {
                        for (int ji = 0; ji < kernelSize; ji++)
                        {
                            int curX = i * kernelSize + ki;
                            int curY = j * kernelSize + ji;

                            if (curX >= originalSize) curX = originalSize - 1;
                            if (curY >= originalSize) curY = originalSize - 1;

                            if (maxCoordinates.Any(q => q.X == curX && q.Y == curY))
                            {
                                output[curX, curY] = input[i, j];
                            }
                            else
                            {
                                output[curX, curY] = 0;
                            }
                        }
                    }
                }
            }

            return output;
        }

        public static double[][,] Convolute(double[][,] input, double[][,] kernels)
        {
            if (input.Length != kernels.Length)
            {
                throw new InvalidOperationException("Number of Kernels must be equal to Number of Feature Maps");
            }

            for (int i = 0; i < input.Length; i++)
            {
                if (kernels[i].GetLength(0) != kernels[i].GetLength(1))
                {
                    throw new InvalidOperationException("Convolutional Kernel dimensions are not valid");
                }

                if (input[i].GetLength(0) < kernels[i].GetLength(0)
                    || input[i].GetLength(1) < kernels[i].GetLength(1))
                {
                    throw new InvalidOperationException("Feature Map must be larger than Kernel");
                }

                input[i] = Convolute(input[i], kernels[i]);
            }

            return input;
        }

        public static double[,] Convolute(double[,] input, double[,] kernel)
        {
            int kernelSize = kernel.GetLength(0);

            int height = input.GetLength(0);
            int width = input.GetLength(1);

            var output = new double[width - kernelSize + 1, height - kernelSize + 1];

            for (int i = 0; i < height - kernelSize + 1; i++)
            {
                for (int j = 0; j < width - kernelSize + 1; j++)
                {
                    double pixel = 0;

                    for (int k = 0; k < kernelSize; k++)
                    {
                        for (int z = 0; z < kernelSize; z++)
                        {
                            double matrixPixel = input[i + k, j + z];
                            double kernelPixel = kernel[k, z];

                            pixel += matrixPixel * kernelPixel;
                        }
                    }

                    output[i, j] = pixel;
                }
            }

            return output;
        }

        public static double[,] Add(double[,] matrix1, double[,] matrix2)
        {
            if (matrix1.GetLength(0) != matrix2.GetLength(0) ||
                matrix1.GetLength(1) != matrix2.GetLength(1))
            {
                throw new Exception("Can not add matrixes with different dimensions.");
            }

            for (int i = 0; i < matrix1.GetLength(0); i++)
            {
                for (int j = 0; j < matrix1.GetLength(1); j++)
                {
                    matrix1[i, j] += matrix2[i, j];
                }
            }

            return matrix1;
        }

        public static double[,,] Flip(double[,,] input)
        {
            int w = input.GetLength(1);
            int h = input.GetLength(2);
            var output = new double[input.GetLength(0), w, h];
            input.ForEach((q,k,i,j) => output[k, w - i - 1, h - j - 1] = q);
            return output;
        }

        public static double[,] Flip(double[,] input)
        {
            int w = input.GetLength(0);
            int h = input.GetLength(1);
            var output = new double[w, h];
            input.ForEach((q, i, j) => output[w - i - 1, h - j - 1] = q);
            return output;
        }

        public static double[,,] Pad(double[,,] input, int padding)
        {
            int d = input.GetLength(0);
            int w = input.GetLength(1) + padding*2;
            int h = input.GetLength(2) + padding*2;

            var output = new double[d, w, h];

            input.ForEach((q, k, i, j) => output[k, i + padding, j + padding] = q);

            return output;
        }

        public static double[,] Pad(double[,] input, int padding)
        {
            int w = input.GetLength(0) + padding * 2;
            int h = input.GetLength(1) + padding * 2;

            var output = new double[w, h];

            input.ForEach((q, i, j) => output[i + padding, j + padding] = q);

            return output;
        }

        public static double[,,] Unpad(double[,,] input, int padding)
        {
            int d = input.GetLength(0);
            int w = input.GetLength(1) - 2 * padding;
            int h = input.GetLength(2) - 2 * padding;

            var output = new double[d, w, h];

            output.ForEach((k,i,j) => output[k,i,j] = input[k, i + padding, j + padding]);

            return output;
        }
    }
}
