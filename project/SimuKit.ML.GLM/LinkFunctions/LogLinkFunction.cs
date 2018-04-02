using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SimuKit.ML.GLM.LinkFunctions
{
    /// <summary>
    /// The poisson link function maps the constraint interval b to linear line a
    /// The inverse poisson link function maps the linear line a to constraint interval b
    /// 
    /// Generally applicable to the following distribution: 
    ///   1. Poisson: count of occurrences in a fixed amount of time/space (Poisson regression)
    /// </summary>
    public class LogLinkFunction : BaseLinkFunction
    {
        public override double GetLink(double b)
        {
            return System.Math.Log(b);
        }

        public override double GetInvLink(double a)
        {
            return System.Math.Exp(a);
        }

        public override double GetInvLinkDerivative(double a)
        {
            return System.Math.Exp(a);
        }
    }
}
