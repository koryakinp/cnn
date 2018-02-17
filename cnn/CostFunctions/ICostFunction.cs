using System;
using System.Collections.Generic;
using System.Text;

namespace Cnn.CostFunctions
{
    /// <summary>
    /// A cost function to be used to calculate an error value the output
    /// </summary>
    public interface ICostFunction
    {
        /// <summary>
        /// Computes an error value of the output
        /// </summary>
        /// <param name="target">Output neuron target value</param>
        /// <param name="output">Output neuron value</param>
        /// <returns>Error value</returns>
        double ComputeValue(double target, double output);


        /// <summary>
        /// Computes the deriviative of the error value of the output
        /// </summary>
        /// <param name="target">Output neuron target value</param>
        /// <param name="output">Output neuron value</param>
        /// <returns>Deriviative value of the error</returns>
        double ComputeDeriviative(double target, double output);
    }
}
