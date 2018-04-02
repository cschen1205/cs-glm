using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SimuKit.Math.LinearAlgebra;
using SimuKit.Math.Statistics;
using System.Diagnostics;
using SimuKit.Math.LinAlg;

namespace SimuKit.ML.GLM
{
    /// <summary>
    /// The implementation of Glm based on IRLS QR Newton variant
    /// The idea is to compute the QR factorization of matrix matrix A once. The factorization is then used in the IRLS iteration
    /// 
    /// QR factorization results in potentially much better numerical stability since no matrix inversion is actually performed
    /// The cholesky factorization used to compute s requires W be positive definite matrix (i.e., z * W * z be positive for any vector z),
    /// The positive definite matrix requirement can be easily checked by examining the diagonal entries of the weight matrix W
    /// </summary>
    public class GlmIrlsQrNewton : Glm
    {
        private IMatrix<int, double> A;
        private IVector<int, double> b;
        private IMatrix<int, double> At;
        private double[] mX;

        public GlmIrlsQrNewton(GlmDistributionFamily distribution, ILinkFunction linkFunc, double[,] A, double[] b)
            : base(distribution, linkFunc, null, null, null)
        {
            this.A = new SparseMatrix<int, double>(A);
            this.b = new SparseVector<int, double>(b);
            this.At = this.A.Transpose();
            this.mStats = new Statistics.GlmStatistics(A.GetLength(1), b.Length);
        }

        public GlmIrlsQrNewton(GlmDistributionFamily distribution, double[,] A, double[] b)
            : base(distribution)
        {
            this.A = new SparseMatrix<int, double>(A);
            this.b = new SparseVector<int, double>(b);
            this.At = this.A.Transpose();
            this.mStats = new Statistics.GlmStatistics(A.GetLength(1), b.Length);
        }

        public override double[] X
        {
            get
            {
                return mX;
            }
        }

        private static Random rand = new Random();
        public override double[] Solve()
        {
            int m = A.RowCount;
            int n = A.ColCount;

            Debug.Assert(m >= n);

            IVector<int, double> s = new SparseVector<int, double>(n);
            IVector<int, double> sy = new SparseVector<int, double>(n);
            for (int i = 0; i < n; ++i)
            {
                s[i] = 0;
            }

            IVector<int, double> t = new SparseVector<int, double>(m);
            for (int i = 0; i < m; ++i)
            {
                t[i] = 0;
            }

            double[] g = new double[m];
            double[] gprime = new double[m];

            IMatrix<int, double> Q;
            IMatrix<int, double> R;
            QR<double>.Factorize(A, out Q, out R); // A is m x n, Q is m x n orthogonal matrix, R is n x n (R will be upper triangular matrix if m == n)
            
            IMatrix<int, double> Qt = Q.Transpose();

            IVector<int, double> W = null;
            for (int j = 0; j < mMaxIters; ++j)
            {
                IVector<int, double> z = new SparseVector<int, double>(m);
                
                for (int k = 0; k < m; ++k)
                {
                    g[k] = mLinkFunc.GetInvLink(t[k]);
                    gprime[k] = mLinkFunc.GetInvLinkDerivative(t[k]);

                    z[k] = t[k] + (b[k] - g[k]) / gprime[k];
                }
                
                W = new SparseVector<int, double>(m);
                double w_kk_min = double.MaxValue;
                for (int k = 0; k < m; ++k)
                {
                    double g_variance = GetVariance(g[k]);
                    double w_kk = gprime[k] * gprime[k] / (g_variance);
                    W[k] = w_kk;
                    w_kk_min = System.Math.Min(w_kk, w_kk_min);
                }

                if (w_kk_min < System.Math.Sqrt(double.Epsilon))
                {
                    Console.WriteLine("Warning: Tiny weights encountered, min(diag(W)) is too small");
                }

                IVector<int, double> s_old = s;

               
                IMatrix<int, double> WQ = new SparseMatrix<int, double>(m, n); // W * Q
                IVector<int, double> Wz = new SparseVector<int, double>(m); // W * z
                for (int k = 0; k < m; k ++)
                {
                    Wz[k] = z[k] * W[k];
                    for (int k2 = 0; k2 < m; ++k2)
                    {
                        WQ[k, k2] = Q[k, k2] * W[k];
                    }
                }
                
                IMatrix<int, double> QtWQ = Qt.Multiply(WQ); // a n x n positive definite matrix, therefore can apply Cholesky
                IVector<int, double> QtWz = Qt.Multiply(Wz);

                IMatrix<int, double> L;
                Cholesky<double>.Factorize(QtWQ, out L);

                IMatrix<int, double> Lt = L.Transpose();

                // (Qt * W * Q) * s = Qt * W * z;
                // L * Lt * s = Qt * W * z (Cholesky factorization on Qt * W * Q)
                // L * sy = Qt * W * z, Lt * s = sy
                // Now forward solve sy for L * sy = Qt * W * z
                // Now backward solve s for Lt * s = sy
                s = new SparseVector<int, double>(n);
                for (int i = 0; i < n; ++i)
                {
                    s[i] = 0;
                    sy[i] = 0;
                }

                //forward solve sy for L * sy = Qt * W * z
                //Console.WriteLine(L);
                for (int i = 0; i < n; ++i)
                {
                    double cross_prod = 0;
                    for (int k = 0; k < i; ++k)
                    {
                        cross_prod += L[i, k] * sy[k];
                    }
                    sy[i] = (QtWz[i] - cross_prod) / L[i, i];
                }
                //backward solve s for U * s = sy
                for (int i = n - 1; i >= 0; --i)
                {
                    double cross_prod = 0;
                    for (int k = i + 1; k < n; ++k)
                    {
                        cross_prod += Lt[i, k] * s[k];
                    }
                    s[i] = (sy[i] - cross_prod) / Lt[i, i];
                }

                t = Q.Multiply(s);

                if ((s_old.Minus(s)).Norm(2) < mTol)
                {
                    break;
                }
            }

            mX = new double[n];
            
            //backsolve x for R * x = Qt * t
            IVector<int, double> c = Qt.Multiply(t);

            for (int i = n - 1; i >= 0; --i) // since m >= n
            {
                double cross_prod = 0;
                for (int j = i + 1; j < n; ++j)
                {
                    cross_prod += R[i, j] * mX[j];
                }
                mX[i] = (c[i] - cross_prod) / R[i, i];
            }

            UpdateStatistics(W);

            return X;

        }

        protected void UpdateStatistics(IVector<int, double> W)
        {
            IMatrix<int, double> AtWA = At.ScalarMultiply(W).Multiply(A);
            IMatrix<int, double> AtWAInv = QRSolver<double>.Invert(AtWA);

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
