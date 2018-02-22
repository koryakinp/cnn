namespace Cnn.Misc
{
    internal class MaxPoolResult
    {
        public double[,] Values { get; set; }
        public Coordinate[] MaxCoordinates { get; set; }
    }

    internal class Coordinate
    {
        public Coordinate(int x, int y)
        {
            X = x;
            Y = y;
        }

        public readonly int X;
        public readonly int Y;
    }
}
