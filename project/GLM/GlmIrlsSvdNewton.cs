using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using cs_matrix;
using ContinuousOptimization.Statistics;

namespace GlmSharp
{
    /// <summary>
    /// The implementation of Glm based on IRLS SVD Newton variant
    /// 
    /// If you are really concerned about the rank deficiency of the model matrix A and the QR New variant does not a rank-revealing QR method 
    /// , then SVD Newton variant IRLS can be used (albeit slower than the QR variant). 
    /// 
    /// The SVD-based method is adapted from the QR variant to used SVD to definitively determines the rank of the model matrix.
    ///  
    /// Note that SVD is also potentially much stable compare to the basic IRLS as it uses pseudo inverse 
    /// Note that SVD is slower if m >> n since it involves  m dimension multiplication and the SVD for large m is costly
    /// </summary>
    public class GlmIrlsSvdNewton : Glm
    {
        private IMatrix A;
        private IVector b;
        private IMatrix At;
        public double epsilon = 1e-10;
        private double[] mX;

        public GlmIrlsSvdNewton(GlmDistributionFamily distribution, ILinkFunction linkFunc, double[,] A, double[] b)
            : base(distribution, linkFunc, null, null, null)
        {
            this.A = new SparseMatrix(A);
            this.b = new SparseVector(b);
            this.At = this.A.Transpose();
            this.mStats = new Statistics.GlmStatistics(A.GetLength(1), b.Length);
        }

        public GlmIrlsSvdNewton(GlmDistributionFamily distribution, double[,] A, double[] b)
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

            Debug.Assert(m >= n);
            
            IVector t = new SparseVector(n);
            for (int i = 0; i < m; ++i)
            {
                t[i] = 0;
            }
            IVector s = new SparseVector(n);
            IVector sy = new SparseVector(n);
            for (int i = 0; i < n; ++i)
            {
                s[i] = 0;
            }
            IVector s_old = s;

            IMatrix U; // U is a m x m orthogonal matrix 
            IMatrix Vt; // V is a n x n orthogonal matrix
            IMatrix Sigma; // Sigma is a m x n diagonal matrix with non-negative real numbers on its diagonal 
            SVD.Factorize(A, out U, out Sigma, out Vt); // A is a m x n matrix
           
            IMatrix Ut = U.Transpose();
            IMatrix V = Vt.Transpose();
           

            //SigmaInv is obtained by replacing every non-zero diagonal entry by its reciprocal and transposing the resulting matrix
            IMatrix SigmaInv = new SparseMatrix(m, n);
            for (int i = 0; i < n; ++i) // assuming m >= n
            {
                double sigma_i = Sigma[i, i];
                if (sigma_i < epsilon) // model matrix A is rank deficient
                {
                    throw new Exception("Near rank-deficient model matrix");
                }
                SigmaInv[i, i] = 1.0 / sigma_i;
            }
            SigmaInv = SigmaInv.Transpose();

            double[] W = new double[m];

            for (int j = 0; j < mMaxIters; ++j)
            {
                Console.WriteLine("j: {0}", j);

                IVector z = new SparseVector(m);
                double[] g = new double[m];
                double[] gprime = new double[m];

                for (int k = 0; k < m; ++k)
                {
                    g[k] = mLinkFunc.GetInvLink(t[k]);
                    gprime[k] = mLinkFunc.GetInvLinkDerivative(t[k]);

                    z[k] = t[k] + (b[k] - g[k]) / gprime[k];
                }

                int tiny_weight_count = 0;
                for (int k = 0; k < m; ++k)
                {
                    double w_kk = gprime[k] * gprime[k] / GetVariance(g[k]);
                    W[k] = w_kk;
                    if (w_kk < double.Epsilon * 2)
                    {
                        tiny_weight_count++;
                    }
                }

                if (tiny_weight_count > 0)
                {
                    Console.WriteLine("Warning: tiny weights encountered, (diag(W)) is too small");
                }

                s_old = s;

                IMatrix UtW = new SparseMatrix(m, m);
                for (int k = 0; k < m; ++k)
                {
                    for (int k2 = 0; k2 < m; ++k2)
                    {
                        UtW[k, k2] = Ut[k, k2] * W[k];
                    }
                }

                IMatrix UtWU = UtW.Multiply(U); // m x m positive definite matrix
                IMatrix L; // m x m lower triangular matrix
                Cholesky.Factorize(UtWU, out L);
                
                IMatrix Lt = L.Transpose(); // m x m upper triangular matrix

                IVector UtWz = UtW.Multiply(z); // m x 1 vector

                // (Ut * W * U) * s = Ut * W * z
                // L * Lt * s = Ut * W * z (Cholesky factorization on Ut * W * U)
                // L * sy = Ut * W * z, Lt * s = sy 
                s = new SparseVector(n);
                for (int i = 0; i < n; ++i)
                {
                    s[i] = 0;
                    sy[i] = 0;
                }

                // forward solve sy for L * sy = Ut * W * z
                for (int i = 0; i < n; ++i)  // since m >= n
                {
                    double cross_prod = 0;
                    for (int k = 0; k < i; ++k)
                    {
                        cross_prod += L[i, k] * sy[k];
                    }
                    sy[i] = (UtWz[i] - cross_prod) / L[i, i];
                }
                // backward solve s for Lt * s = sy
                for (int i = n - 1; i >= 0; --i) 
                {
                    double cross_prod = 0;
                    for (int k = i + 1; k < n; ++k)
                    {
                        cross_prod += Lt[i, k] * s[k];
                    }
                    s[i] = (sy[i] - cross_prod) / Lt[i, i];
                }


                t = U.Multiply(s);

                if ((s_old.Minus(s)).Norm(2) < mTol)
                {
                    break;
                }
            }

            IVector x = V.Multiply(SigmaInv).Multiply(Ut).Multiply(t);

            mX = new double[n];
            for (int i = 0; i < n; ++i)
            {
                mX[i] = x[i];
            }

            UpdateStatistics(W);

            return X;
        }

        public override double[] X
        {
            get
            {
                return mX;
            }
        }

        protected void UpdateStatistics(double[] W)
        {
            IMatrix AtWA = At.ScalarMultiply(W).Multiply(A);
            IMatrix AtWAInv = QRSolver.Invert(AtWA);

            int n = AtWAInv.RowCount;
            int m = b.Dimension;

            for (int i = 0; i < n; ++i)
            {
                mStats.StandardErrors[i] = System.Math.Sqrt(AtWAInv[i, i]);
                for (int j = 0; j < n; ++j)
                {
                    mStats.VCovMatrix[i][j] = AtWAInv[i, j];
                }
            }

            double[] outcomes = new double[m];
            for (int i = 0; i < m; i++)
            {
                double cross_prod = 0;
                for (int j = 0; j < n; ++j)
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
    }
}
