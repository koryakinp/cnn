using Cnn.Misc;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Cnn
{
    internal static class MatrixProcessor
    {
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
    }
}
