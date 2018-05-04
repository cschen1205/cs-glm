using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using ContinuousOptimization.Statistics;
using cs_matrix;

namespace GlmSharp.Statistics
{
    public class GlmStatistics
    {
        protected double[][] mVcovMatrix;
        protected double[] mResiduals;
        protected double mResidualStdDev;
        protected double[] mStandardErrors;
        protected double mAdjustedR2;
        protected double mR2;
        protected double mResponseVariance;
        protected double mResponseMean;

        public GlmStatistics()
        {

        }

        /// <summary>
        /// In this particular instance, it is assumed that W = sigma^(-2) I, that is e is identical and uncorrelated
        /// </summary>
        /// <param name="A"></param>
        /// <param name="b"></param>
        /// <param name="x"></param>
        public GlmStatistics(double[][] A, double[] b, double[] x)
        {
            int m = A.GetLength(0);
            int n = A.GetLength(1);

            mResiduals = new double[m];                    

            for (int i = 0; i < m; ++i)
            {
                double cross_prod = 0;
                for (int j = 0; j < n; ++j)
                {
                    cross_prod += A[i][j] * x[j];
                }
                mResiduals[i] = b[i] - cross_prod;
            }

            double residual_mu = 0;
            mResidualStdDev = StdDev.GetStdDev(mResiduals, residual_mu);

            mResponseMean = Mean.GetMean(b);
            mResponseVariance = System.Math.Pow(StdDev.GetStdDev(b, mResponseMean), 2);

            // (A.transpose * A).inverse * sigma^2
            IMatrix AtA = new SparseMatrix(n, n);
            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    double cross_prod = 0;
                    for (int k = 0; k < m; ++k)
                    {
                        cross_prod += A[i][k] * A[k][j];
                    }
                    AtA[i, j] = cross_prod;
                }
            }
            IMatrix AtAInv = QRSolver.Invert(AtA);
            double sigmaSq = mResidualStdDev * mResidualStdDev;

            mVcovMatrix = new double[n][];
            for (int i = 0; i < n; ++i)
            {
                mVcovMatrix[i] = new double[n];
                for (int j = 0; j < n; ++j)
                {
                    mVcovMatrix[i][j] = AtAInv[i, j] * sigmaSq;
                }
            }

            mStandardErrors = new double[n];
            for (int i = 0; i < n; ++i)
            {
                mStandardErrors[i] = System.Math.Sqrt(mVcovMatrix[i][i]);
            }

            mR2 = 1 - sigmaSq / mResponseVariance;
            mAdjustedR2 = 1 - sigmaSq / mResponseVariance * (n - 1) / (n - x.Length - 1);

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="n">Number of variables</param>
        /// <param name="m">Number of data points</param>
        public GlmStatistics(int n, int m)
        {
            mVcovMatrix = new double[n][];
            for (int i = 0; i < n; ++i)
            {
                mVcovMatrix[i] = new double[n];
            }

            mResiduals = new double[m];
            mStandardErrors = new double[n];
        }

        /// <summary>
        /// variance-covariance matrix
        /// </summary>
        public double[][] VCovMatrix
        {
            get
            {
                return mVcovMatrix;
            }
        }

        public double[] Residuals
        {
            get
            {
                return mResiduals;
            }
        }

        public double ResidualStdDev
        {
            get
            {
                return mResidualStdDev;
            }
            set
            {
                mResidualStdDev = value;
            }
        }

        public double RSS
        {
            get
            {
                return mResidualStdDev* mResidualStdDev;
            }
        }

        public double[] StandardErrors
        {
            get
            {
                return mStandardErrors;
            }
        }

        public double AdjustedR2
        {
            get
            {
                return mAdjustedR2;
            }
            set
            {
                mAdjustedR2 = value;
            }
        }

        public double R2
        {
            get
            {
                return mR2;
            }
            set
            {
                mR2 = value;
            }
        }

        public double ResponseVariance
        {
            get
            {
                return mResponseVariance;
            }
            set
            {
                mResponseVariance = value;
            }
        }

        public double ResponseMean
        {
            get
            {
                return mResponseMean;
            }
            set
            {
                mResponseMean = value;
            }
        }
    }
}
