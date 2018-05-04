using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using ContinuousOptimization.Statistics;

namespace GlmSharp
{
    /// <summary>
    /// The likelihood function for the linear model
    /// 
    /// </summary>
    public class GlmLikelihoodFunction
    {
        /// <summary>
        /// Return the likelihood value of the fitted regression model
        /// </summary>
        /// <param name="data"></param>
        /// <param name="beta_hat">estimated predictor coefficient in the fitted regression model</param>
        /// <returns></returns>
        public static double GetLikelihood(GlmDistributionFamily distribution, List<RDataRecord> data, double[] beta_hat)
        {
            switch (distribution)
            {
                case GlmDistributionFamily.Normal:
                    return GetLikelihood_Normal(data, beta_hat);
                default:
                    throw new NotImplementedException();
            }
        }

        private static double GetLikelihood_Normal(List<RDataRecord> data,double[] beta_hat)
        {
            int N = data.Count;
            int k = beta_hat.Length;
            double residual_sum_of_squares = 0;

            double[] y = new double[N];
            for(int i=0; i < N; ++i)
            {
                y[i] = data[i].YValue;
            }

            double sigma = StdDev.GetStdDev(y, Mean.GetMean(y)) / (N - k - 1);

            for (int i = 0; i < N; ++i)
            {
                double linear_predictor = 0;
                RDataRecord rec = data[i];
                for (int j = 0; j < k; ++j)
                {
                    linear_predictor += rec.data[j] * beta_hat[j];
                }
                double residual = rec.YValue - linear_predictor;
                residual_sum_of_squares += residual*residual;
            }

            return System.Math.Exp(-residual_sum_of_squares / (2 * sigma)) / System.Math.Sqrt(2 * System.Math.PI * sigma);
        }
    }
}
