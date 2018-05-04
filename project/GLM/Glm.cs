using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using GlmSharp.LinkFunctions;
using System.Diagnostics;
using GlmSharp.Statistics;
using ContinuousOptimization;
using ContinuousOptimization.LinearAlgebra;

/// <summary>
/// In regression, we tried to find a set of model coefficient such for 
/// A * x = b + e
/// A * x is known as the model matrix, b as the response vector, e is the error terms.
/// 
/// The description below is for OLS for the form:
/// A * x = b + e
/// In GLM, A * x is replaced by g(A * x) where g() is an inverse link function associated with the distribution of b (in other words, it maps the linear line A * x to constraint interval b), therefore we have:
/// g(A * x) = b + e
/// The discussion that follows are derived based on OLS, but applicable to GLM as well with some modification
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
/// dy / dx = A.transpose * W * (A * x - b) * 2 + (A * x - b).transpose * (dW / dx) * (A * x - b)
/// 
/// To find min y, set dy / dx = 0 at x = x_bar, we have
/// A.transpose * W * (A * x_bar - b) = 0 
/// The above assumes that dW / dx = 0 for x = x_bar, as the variance-covariance matrix is minimized at x = x_bar
/// 
/// Transform this, we have
/// A.transpose * W * A * x_bar = A.transpose * W * b
/// 
/// Multiply both side by (A.transpose * W * A).inverse, we have
/// x_bar = (A.transpose * W * A).inverse * A.transpose * W * b
/// This is commonly solved using IRLS
/// 
/// Suppose we assumes that e consist of uncorrelated random variables with identical variance, then W = sigma^(-2) * I, then x_bar becomes
/// x_bar = (A.transpose * A).inverse * A.transpose * b 
/// for min (A * x - b).transpose * (A * x -b)
/// This can be solved using the default solver implemented in Glm class
/// </summary>
namespace GlmSharp
{
    /// <summary>
    /// Link: http://bwlewis.github.io/GLM/
    /// GLM is generalized linear model for exponential family of distribution model b = g(a).
    /// g(a) is the inverse link function. 
    /// 
    /// Therefore, for a regression characterized by inverse link function g(a), the regression problem be formulated
    /// as we are looking for model coefficient set x in 
    /// g(A * x) = b + e
    /// And the objective is to find x such for the following objective:
    /// min (g(A * x) - b).transpose * W * (g(A * x) - b)
    /// 
    /// Suppose we assumes that e consist of uncorrelated random variables with identical variance, then W = sigma^(-2) * I, 
    /// and The objective min (g(A * x) - b) * W * (g(A * x) - b).tranpose is reduced to the OLS form:
    /// min || g(A * x) - b ||^2
    /// </summary>
    public class Glm
    {
        private SingleTrajectoryContinuousSolver solver;
        protected ILinkFunction mLinkFunc;
        private double[][] A; //first column of A corresponds to x_0 = 1
        private double[] b;
        protected int mMaxIters = 25;
        protected double mTol = 0.000001;
        protected double mRegularizationLambda = 0;
        private double[] mX;
        protected GlmDistributionFamily mDistributionFamily;
        protected GlmStatistics mStats = new GlmStatistics();
        
        public double Tol
        {
            get { return mTol; }
            set { mTol = value; }
        }

        public GlmDistributionFamily DistributionFamily
        {
            get { return mDistributionFamily; }
        }

        public double Predict(double[] input_0)
        {
            int n = input_0.Length;
            double[] coef = X;
            Debug.Assert(n == coef.Length);
            double linear_predictor = 0;
            for (int i = 0; i < n; ++i)
            {
                linear_predictor += coef[i] * input_0[i];
            }
            return mLinkFunc.GetInvLink(linear_predictor);
        }

       

        protected double GetVariance(double g)
        {
            switch (mDistributionFamily)
            {
                case GlmDistributionFamily.Bernouli:
                case GlmDistributionFamily.Binomial:
                case GlmDistributionFamily.Categorical:
                case GlmDistributionFamily.Multinomial:
                    return g * (1-g);
                case GlmDistributionFamily.Exponential:
                case GlmDistributionFamily.Gamma:
                    return g * g;
                case GlmDistributionFamily.InverseGaussian:
                    return g * g * g;
                case GlmDistributionFamily.Normal:
                    return 1;
                case GlmDistributionFamily.Poisson:
                    return g;
                default:
                    throw new NotImplementedException();
            }
        }

        public int MaxIters
        {
            get
            {
                return mMaxIters; 
            }
            set
            {
                mMaxIters = value;
            }
        }

        public virtual double[] X
        {
            get
            {
                return mX;
            }
        }

        public GlmStatistics Statistics
        {
            get
            {
                return mStats;
            }
        }

        public Glm(GlmDistributionFamily distribution, ILinkFunction linkFunc, double[][] A, double[] b, SingleTrajectoryContinuousSolver solver)
        {
            this.mDistributionFamily = distribution;
            this.solver = solver;
            this.mLinkFunc = linkFunc;
            this.A = A;
            this.b = b;
            this.mStats = new Statistics.GlmStatistics(A[0].Length, b.Length);
        }

        public Glm(GlmDistributionFamily distribution, double[][] A, double[] b, SingleTrajectoryContinuousSolver solver)
        {
            this.solver = solver;
            this.mDistributionFamily = distribution;
            this.mLinkFunc = GetLinkFunction(distribution);
            this.A = A;
            this.b = b;
            this.mStats = new Statistics.GlmStatistics(A[0].Length, b.Length);
        }

        public Glm(GlmDistributionFamily distribution)
        {
            this.mLinkFunc = GetLinkFunction(distribution);
            this.mDistributionFamily = distribution;
        }

        public Glm(GlmDistributionFamily distribution, double[,] A, double[] b, SingleTrajectoryContinuousSolver solver, int maxIters = -1)
        {
            this.solver = solver;
            this.mDistributionFamily = distribution;
            this.mLinkFunc = GetLinkFunction(distribution);
            this.A = new double[A.GetLength(0)][];
            for (int i = 0; i < A.GetLength(0); i++)
            {
                this.A[i] = new double[A.GetLength(1)];
                for (int j = 0; j < A.GetLength(1); j++)
                {
                    this.A[i][j] = A[i, j];
                }
            }
            this.b = b;
            if (maxIters > 0)
            {
                this.mMaxIters = maxIters;
            }
            this.mStats = new Statistics.GlmStatistics(A.GetLength(1), b.Length);
        }

        public static ILinkFunction GetLinkFunction(GlmDistributionFamily distribution)
        {
            switch (distribution)
            {
                case GlmDistributionFamily.Bernouli:
                case GlmDistributionFamily.Binomial:
                case GlmDistributionFamily.Categorical:
                case GlmDistributionFamily.Multinomial:
                    return new LogitLinkFunction();
                case GlmDistributionFamily.Exponential:
                case GlmDistributionFamily.Gamma:
                    return new InverseLinkFunction();
                case GlmDistributionFamily.InverseGaussian:
                    return new InverseSquaredLinkFunction();
                case GlmDistributionFamily.Normal:
                    return new IdentityLinkFunction();
                case GlmDistributionFamily.Poisson:
                    return new LogLinkFunction();
                default:
                    throw new NotImplementedException();
            }
        }

        protected double EvaluateCost(double[] x, double[] lower_bounds, double[] upper_bounds, object constraints)
        {
            int m = b.Length;
            int n = x.Length;

            double[] c = MatrixOp.Multiply(A, x);
            double crossprod = 0;
            for (int i = 0; i < m; ++i)
            {
                double g = mLinkFunc.GetInvLink(c[i]);
                double gprime = mLinkFunc.GetInvLinkDerivative(c[i]);

                double d = g - b[i];
                crossprod += d * d;
            }

            double J = crossprod / (2 * m);

            for (int j = 1; j < n; ++j)
            {
                J += (mRegularizationLambda * x[j] * x[j]) / (2 * m);
            }

            return J;
        }

        protected void EvaluateGradient (double[] x, double[] gradx, double[] lower_bounds, double[] upper_bounds, object constraints)
        {
            int m = b.Length;
            int n = A[0].Length;

            double[] c = MatrixOp.Multiply(A, x);

            double[] g = new double[m];
            double[] gprime = new double[m];
            for (int j = 0; j < m; ++j)
            {
                g[j] = mLinkFunc.GetInvLink(c[j]);
                gprime[j] = mLinkFunc.GetInvLinkDerivative(c[j]);
            }
           
            for (int i = 0; i < n; ++i)
            {
                double crossprod = 0;
                for (int j = 0; j < m; ++j)
                {
                    double cb = g[j] - b[j];
                    crossprod += cb * gprime[j] * A[j][i];
                }

                gradx[i] = crossprod / m;

                if (i != 0)
                {
                    gradx[i] += (mRegularizationLambda * x[i]) / m;
                }
            }
            /*
            GradientEstimation.CalcGradient(x, gradx, (x2, constraints2) =>
                {
                    return EvaluateCost(x2, lower_bounds, upper_bounds, constraints2);
                });*/
        }

        public virtual double[] Solve()
        {
            int n = A[0].Length;

            Random random = new Random();
            double[] x_0 = new double[n];
            for (int i = 0; i < n; ++i)
            {
                x_0[i] = random.NextDouble();
            }
            ContinuousSolution s = solver.Minimize(x_0, EvaluateCost, EvaluateGradient, ShouldTerminate);

            mX = s.Values;

            UpdateStatistics();

            return X;
        }

        private void UpdateStatistics()
        {
            mStats = new GlmStatistics(A, b, mX);
        }

        protected bool ShouldTerminate(double? improvement, int iterations)
        {
            if (improvement.HasValue && improvement.Value < mTol)
            {
                return false;
            }
            return iterations >= mMaxIters;
        }
    }
}
