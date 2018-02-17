using cnn;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Cnn.Layers
{
    internal class BorderLayer : Layer
    {
        private double[][,] _featureMaps { get; set; }

        public BorderLayer(int layerIndex) : base(layerIndex) {}

        public override Value PassBackward(Value value)
        {
            var res = value
                .Single
                .ToList()
                .SplitBySize(value.Single.Length/_featureMaps.Length)
                .Select(q => q.ToArray().SingleToDouble(_featureMaps[0].GetLength(0)))
                .ToArray();

            return new MultiValue(res);
        }

        public override Value PassForward(Value value)
        {
            if(_featureMaps == null)
            {
                _featureMaps = new double[value.Multi.Length][,];
                value.Multi.ForEach((q,i) => _featureMaps[i] = new double[q.GetLength(0),q.GetLength(1)]);
            }

            double[] val = value.Multi.SelectMany(q => q.Cast<double>()).ToArray();
            return new SingleValue(val);
        }
    }
}
