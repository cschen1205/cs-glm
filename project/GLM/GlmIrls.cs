using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using ContinuousOptimization.Statistics;
using cs_matrix;

/// <summary>
/// In regression, we tried to find a set of model coefficient such for 
/// A * x = b + e
/// 
/// A * x is known as the model matrix, b as the response vector, e is the error terms.
/// 
/// In OLS (Ordinary Least Square), we assumes that the variance-covariance matrix V(e) = sigma^2 * W, where:
///   W is a symmetric positive definite matrix, and is a diagonal matrix
///   sigma is the standard error of e
/// 
/// In OLS (Ordinary Least Square), the objective is to find x_bar such that e.tranpose * W * e is minimized (Note that since W is positive definite, e * W * e is alway positive)
/// In other words, we are looking for x_bar such as (A * x_bar - b).transpose * W * (A * x_bar - b) is minimized
/// 
/// Let y = (A * x - b).transpose * W * (A * x - b)
/// Now differentiating y with respect to x, we have
/// dy / dx = A.transpose * W * (A * x - b) * 2
/// 
/// To find min y, set dy / dx = 0 at x = x_bar, we have
/// A.transpose * W * (A * x_bar - b) = 0
/// 
/// Transform this, we have
/// A.transpose * W * A * x_bar = A.transpose * W * b
/// 
/// Multiply both side by (A.transpose * W * A).inverse, we have
/// x_bar = (A.transpose * W * A).inverse * A.transpose * W * b
/// This is commonly solved using IRLS
/// </summary>
namespace GlmSharp
{
    /// <summary>
    /// The implementation of Glm based on iteratively reweighted least squares estimation (IRLS)
    /// 
    /// Discussion:
    /// 
    /// As inversion is performed for A.transpose * W * A, since A.transpose * W * A may not be directly invertable, the IRLS in this implementation is potentially numerically
    /// unstable and not generally advised, use the QR or SVD variant of IRLS instead
    /// 
    /// </summary>
    public class GlmIrls : Glm
    {
        private IMatrix A;
        private IVector b;
        private IMatrix At;
        private double[] mX;
        
        public GlmIrls(GlmDistributionFamily distribution, ILinkFunction linkFunc, double[,] A, double[] b)
            : base(distribution, linkFunc, null, null, null)
        {
            this.A = new SparseMatrix(A);
            this.b = new SparseVector(b);
            this.At = this.A.Transpose();
            this.mStats = new Statistics.GlmStatistics(A.GetLength(1), b.Length);
        }

        public GlmIrls(GlmDistributionFamily distribution, double[,] A, double[] b)
            : base(distribution)
        {
            this.A = new SparseMatrix(A);
            this.b = new SparseVector(b);
            this.At = this.A.Transpose();
            this.mStats = new Statistics.GlmStatistics(A.GetLength(1), b.Length);
        }

        public override double[] Solve()
        {
            int m = A.RowCount;
            int n = A.ColCount;
            
            IVector x = new SparseVector(n);
            for (int i = 0; i < n; ++i)
            {
                x[i] = 0;
            }

            IMatrix W = null;
            IMatrix AtWAInv = null;

            for (int j = 0; j < mMaxIters; ++j)
            {
                IVector eta = A.Multiply(x);
                IVector z = new SparseVector(m);
                double[] g = new double[m];
                double[] gprime = new double[m];

                for (int k = 0; k < m; ++k)
                {
                    g[k] = mLinkFunc.GetInvLink(eta[k]);
                    gprime[k] = mLinkFunc.GetInvLinkDerivative(eta[k]);

                    z[k] = eta[k] + (b[k] - g[k]) / gprime[k];
                }

                W = A.Identity(m);
                for (int k = 0; k < m; ++k)
                {
                    double g_variance = GetVariance(g[k]);
                    if(g_variance==0)
                    {
                        Environment.Exit(0);
                    }
                    W[k, k] = gprime[k] * gprime[k] / g_variance;
                }

                IVector x_old = x;

                IMatrix AtW = At.Multiply(W);

                // solve x for At * W * A * x = At * W * z
                IMatrix AtWA = AtW.Multiply(A);
                AtWAInv = QRSolver.Invert(AtWA);
                x = AtWAInv.Multiply(AtW).Multiply(z);

                double cost = x.Minus(x_old).Norm(2);
                if (j % 10 == 0)
                {
                    Console.WriteLine("Iteration: {0}, Cost: {1}", j, cost);
                }

                if (cost < mTol)
                {
                    break;
                }
            }

            mX = new double[n];
            for (int i = 0; i < n; ++i)
            {
                mX[i] = x[i];
            }

            UpdateStatistics(AtWAInv, W);

            return mX;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="vcovmat">variance-covariance matrix for the model coefficients</param>
        private void UpdateStatistics(IMatrix vcovmat, IMatrix W)
        {
            int n = vcovmat.RowCount;
            int m = b.Dimension;

            for (int i = 0; i < n; ++i)
            {
                mStats.StandardErrors[i] = System.Math.Sqrt(vcovmat[i, i]);
                for (int j = 0; j < n; ++j)
                {
                    mStats.VCovMatrix[i][j] = vcovmat[i, j];
                }
            }

            double[] outcomes = new double[m];
            for (int i = 0; i < m; i++)
            {
                double cross_prod = 0;
                for(int j=0; j < n; ++j)
                {
                    cross_prod += A[i, j] * mX[j];
                }
                mStats.Residuals[i] = b[i] - mLinkFunc.GetInvLink(cross_prod);
                outcomes[i] = b[i];
            }

            mStats.ResidualStdDev = StdDev.GetStdDev(mStats.Residuals, 0);
            mStats.ResponseMean = Mean.GetMean(outcomes);
            mStats.ResponseVariance = System.Math.Pow(StdDev.GetStdDev(outcomes, mStats.ResponseMean), 2);

            mStats.R2 = 1 - mStats.ResidualStdDev * mStats.ResidualStdDev / mStats.ResponseVariance;
            mStats.AdjustedR2 = 1 - mStats.ResidualStdDev * mStats.ResidualStdDev / mStats.ResponseVariance * (n - 1) / (n - mX.Length - 1);
        }

        public override double[] X
        {
            get
            {
                return mX;
            }
        }
    }
}
