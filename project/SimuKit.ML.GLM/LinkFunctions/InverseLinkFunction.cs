using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SimuKit.ML.GLM.LinkFunctions
{
    /// <summary>
    /// Generally applicable to the following distribution:
    ///   1. Exponential / Gamma : Exponential-response data, scale parameters
    /// </summary>
    public class InverseLinkFunction : BaseLinkFunction
    {
        public override double GetLink(double b)
        {
            return -1.0 / b;
        }

        public override double GetInvLink(double a)
        {
            return -1.0 / a;
        }

        public override double GetInvLinkDerivative(double a)
        {
            return -1.0 / (a * a);
        }
    }
}
