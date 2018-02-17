using System;
using System.Collections.Generic;
using System.Text;

namespace Cnn.CostFunctions
{
    internal static class CostFunctionFactory
    {
        public static ICostFunction Produce(CostFunctionType type)
        {
            switch(type)
            {
                case CostFunctionType.Quadratic: return new QuadraticCostFunction();
                case CostFunctionType.SoftMax: return new SoftMaxCostFunction();
                case CostFunctionType.CrossEntropy: return new CrossEntropyCostFunction();
                default: throw new Exception("Cost function is not supported");
            }
        }
    }
}
