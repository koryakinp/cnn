using System;
using System.Collections.Generic;
using System.Text;

namespace Cnn.LearningRateAnnealers
{
    internal interface ILearningRateAnnealer
    {
        double Compute(double dx);
    }
}
