using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GlmSharp
{
    public class RDataRecord
    {
        public RDataRecord()
        {

        }
        public RDataRecord(int dim)
        {
            data = new double[dim];
        }

        public double[] data { get; set; }
        public double YValue { get; set; }
        public int FeatureCount { get { return data.Length;  } }
        public int Dimension { get { return FeatureCount; } }
    }
}
