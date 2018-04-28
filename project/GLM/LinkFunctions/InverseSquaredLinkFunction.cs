using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SimuKit.ML.GLM.LinkFunctions
{
    /// <summary>
    /// Generally applicable to:
    ///   1. Inverse Gaussian
    /// </summary>
    public class InverseSquaredLinkFunction : BaseLinkFunction
    {
        public override double GetLink(double b)
        {
            return - 1.0 / (b * b);
        }

        public override double GetInvLink(double a)
        {
            return System.Math.Sqrt(-a);
        }

        public override double GetInvLinkDerivative(double a)
        {
            return -1.0 / System.Math.Sqrt(-a);
        }
    }
}
