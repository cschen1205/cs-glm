using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace GlmSharp.LinkFunctions
{
    /// <summary>
    /// For the linear link function:
    /// The constraint interval is a real line as well
    /// 
    /// The linear link function maps constraint interval { b | b \in R } to real line {a | a \in R} by:
    /// a = f(b) = b
    /// 
    /// The inverse link function maps real line {a | a \in R} to constraint interval { b | b \in R } by:
    /// b = g(a) = a
    /// 
    /// Generally applicable to the following distribution:
    ///   1. Normal: which is the linear-response data (linear regression)
    /// </summary>
    public class IdentityLinkFunction : BaseLinkFunction
    {
        /// <summary>
        /// The linear link function maps constraint interval { b | b \in R } to real line {a | a \in R} by:
        /// a = f(b) = b
        /// </summary>
        /// <param name="b">The constraint interval value</param>
        /// <returns>The mapped linear line value</returns>
        public override double GetLink(double b)
        {
            return b;
        }

        /// <summary>
        /// The inverse link function maps real line {a | a \in R} to constraint interval { b | b \in R } by:
        /// b = g(a) = a
        /// </summary>
        /// <param name="a">The linear line value</param>
        /// <returns>The mapped constraint interval value</returns>
        public override double GetInvLink(double a)
        {
            return a;
        }

        public override double GetInvLinkDerivative(double real_line_value)
        {
            return 1;
        }
    }
}
