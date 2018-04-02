using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SimuKit.ML.GLM.LinkFunctions
{
    public abstract class BaseLinkFunction : ILinkFunction
    {
        public abstract double GetLink(double constraint_interval_value);
        public abstract double GetInvLink(double real_line_value);
        public abstract double GetInvLinkDerivative(double real_line_value);
    }
}
