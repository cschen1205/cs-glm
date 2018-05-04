using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace GlmSharp
{
    /// <summary>
    /// A link function a = f(b) is a bijection chosen to map the constraint interval onto the real line.
    /// 
    /// The inverse link function b = g(a) maps the real line onto the constraint interval
    /// </summary>
    public interface ILinkFunction
    {
        /// <summary>
        /// The link function f(b) map the constraint interval {b} onto the real line {a}
        /// </summary>
        /// <param name="constraint_interval_value"></param>
        /// <returns></returns>
        double GetLink(double constraint_interval_value);

        /// <summary>
        /// The inverse link function g(a) Map the real line {a} onto the constraint interval {b}
        /// </summary>
        /// <param name="real_line_value"></param>
        /// <returns></returns>
        double GetInvLink(double real_line_value);

        /// <summary>
        /// Return the derivative of the inverse link function g'(x)
        /// </summary>
        /// <param name="real_line_value"></param>
        /// <returns></returns>
        double GetInvLinkDerivative(double real_line_value);
    }
}
