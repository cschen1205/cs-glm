using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace GlmSharp
{
    public class GlmSolverFactory
    {
        protected GlmDistributionFamily mDistributionFamily = GlmDistributionFamily.Normal;

        public GlmDistributionFamily DistributionFamily
        {
            get { return mDistributionFamily; }
            set { mDistributionFamily = value; }
        }

        private static void Analyze(IList<RDataRecord> records, out Dictionary<int, Dictionary<int, int>> levelOrg, out int n)
        {
            RDataRecord rec = records[0];
            int m = records.Count;
            n = 1;
            levelOrg = new Dictionary<int, Dictionary<int, int>>();
            for (int i = 1; i < n; ++i)
            {
                n++;
            }
        }

        public virtual Glm CreateSolver(IList<RDataRecord> records)
        {
            RDataRecord rec = records[0];
            int d = rec.Dimension;
            int m = records.Count;

            int n = 0;
            Dictionary<int, Dictionary<int, int>> levelOrg;
            Analyze(records, out levelOrg, out n);

            double[,] A = new double[m, n];
            double[] b = new double[m];

            for (int i = 0; i < m; ++i)
            {
                rec = records[i];
                int index = 0;
                for (int j = 0; j < d; ++j)
                {
                    A[i, index++] = rec.data[j];
                }
                b[i] = rec.YValue;
            }

            return new GlmIrlsQrNewton(mDistributionFamily, A, b);
        }

        public static double Predict(Glm solver, IList<RDataRecord> records, RDataRecord input_0)
        {
            int d = input_0.Dimension;

            int n = 0;
            Dictionary<int, Dictionary<int, int>> levelOrg;
            Analyze(records, out levelOrg, out n);

            double[] x = new double[n];
            
            int index = 0;
            for (int j = 0; j < d; ++j)
            {
                x[index++] = input_0.data[j];
            }

            return solver.Predict(x);
        }
    }
}
