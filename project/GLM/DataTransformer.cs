using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GlmSharp
{
    public class DataTransformer
    {
        public void DoFeaturesScaling(List<RDataRecord> records)
        {
            int dim = records[0].data.Length;
            double[] mean = new double[dim];
            double[] stdDev = new double[dim];
            
            for(int d=0; d < dim; ++d)
            {
                double _mean = 0;
                double _stdDev = 0;
                for (int i = 0; i < records.Count; ++i)
                {
                    _mean += records[i].data[d];
                }
                _mean /= records.Count;
                mean[d] = _mean;
                for(int i=0; i < records.Count; ++i)
                {
                    double dif = records[i].data[d] - _mean;
                    _stdDev += dif * dif;
                }
                _stdDev = Math.Sqrt(_stdDev) / (records.Count-1);
                stdDev[d] = _stdDev;
            }

            for(int i=0; i < records.Count; ++i)
            {
                for(int d=0; d < dim; ++d)
                {
                    records[i].data[d] = records[i].data[d] - mean[d] / stdDev[d];
                }
            }
            
        }
    }
}
