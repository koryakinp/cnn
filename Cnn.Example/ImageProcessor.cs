using SixLabors.ImageSharp;
using System;
using System.IO;

namespace Cnn.Example
{
    public static class ImageProcessor
    {
        public static void SaveImage(string path, double[,] data)
        {
            int height = data.GetLength(1);
            int width = data.GetLength(0);

            using (Image<Rgba32> image = new Image<Rgba32>(height, width))
            {
                for (int i = 0; i < height; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        byte val = (byte)Math.Round(data[i, j] * 255);
                        image[i, j] = new Rgba32(val, val, val);
                    }
                }

                image.SaveAsJpeg(File.OpenWrite(path));
            }
        }
    }
}
