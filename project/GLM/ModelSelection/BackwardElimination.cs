﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SimuKit.Solvers.Continuous.LocalSearch;
using SimuKit.Math.Distribution;
using SimuKit.Math.Statistics;
using SimuKit.ML.Lang;

/// <summary>
/// The best model is not always the most complicated. Sometimes including variables that are not
/// evidently important can actually reduce the accuracy of the predictions. The model selection strategies
/// in this case help us to eliminate from the model variables that are less important.
/// 
/// The model that includes all available explanatory variables is often referred to as full model. The goals of
/// model selection is to assess whether the full model is the best model. If it isn't, we want to identify a 
/// smaller model that is preferable.
/// </summary>
namespace GlmSharp.ModelSelection
{
    /// <summary>
    /// The backward-elimination strategy starts with the model that includes all potentiall predictor variables.
    /// Variables are eliminated one-at-a-time from the model until only variables with statistically significant p-values remains.
    /// 
    /// </summary>
    public class BackwardElimination
    {
        /// <summary>
        /// The strategy within each elimination step is to drop the variable with the largest p-value, refit the model, and reasses
        /// the inclusion of all variables.
        /// </summary>
        /// <param name="solver"></param>
        /// <param name="records"></param>
        /// <param name="significance_level"></param>
        /// <param name="one_sided"></param>
        /// <returns>The index list of the predictor variables to be included in the regression model</returns>
        public static List<int> EliminateByPValue(GlmSolverFactory solverFactory, List<RDataRecord> records, double significance_level = 0.05, bool one_sided = false)
        {
            int full_model_feature_count = records[0].FeatureCount;

            List<int> candidate_features = new List<int>();
            for (int i = 1; i <= full_model_feature_count; ++i)
            {
                candidate_features.Add(i);
            }

            Glm solver = FitModel(solverFactory, records);

            double[] pValues = CalcPValues(solver.X, solver.Statistics.StandardErrors, records.Count, one_sided);

            double maxPValue;
            int featureIndexWithMaxPValue = SelectFeatureIndexWithMaxPValue(pValues, out maxPValue);
            int eliminatedFeatureId = candidate_features[featureIndexWithMaxPValue];
            while (maxPValue > significance_level)
            {
                candidate_features.Remove(eliminatedFeatureId);

                RefitModel(candidate_features, solverFactory, records, out solver);

                pValues = CalcPValues(solver.X, solver.Statistics.StandardErrors, records.Count, one_sided);
                featureIndexWithMaxPValue = SelectFeatureIndexWithMaxPValue(pValues, out maxPValue);
                eliminatedFeatureId = candidate_features[featureIndexWithMaxPValue];
            }
            return candidate_features;
        }

        /// <summary>
        /// This is an alternative to using p-values in the backward elimination by using adjusted R^2.
        /// At each elimination step, we refit the model without each of the variable up for potential eliminiation.
        /// If one of these smaller models has a higher adjusted R^2 than our current model, we pick the smaller 
        /// model with the largest adjusted R^2. We continue in this way until removing variables does not increase adjusted R^2.
        /// </summary>
        /// <param name="solver"></param>
        /// <param name="records"></param>
        /// <returns>The index list of the predictor variables to be included in the regression model</returns>
        public static List<int> BackwardEliminate(GlmSolverFactory solverFactory, List<RDataRecord> records, ModelSelectionCriteria criteria = ModelSelectionCriteria.AdjustedRSquare)
        {
            int full_model_feature_count = records[0].FeatureCount;

            List<int> candidate_features = new List<int>();
            for (int i = 1; i <= full_model_feature_count; ++i)
            {
                candidate_features.Add(i);
            }

            int n = records.Count;

            double[] outcomes = new double[n];
            for (int i = 0; i < n; ++i)
            {
                outcomes[i] = records[i].YValue;
            }

            Glm solver = FitModel(solverFactory, records);

            double fitness_score = -1;
            if (criteria == ModelSelectionCriteria.AdjustedRSquare)
            {
                fitness_score = CalcAdjustedRSquare(solver.Statistics.Residuals, outcomes, candidate_features.Count, records.Count);
            }
            else if (criteria == ModelSelectionCriteria.AIC)
            {
                double L = GlmLikelihoodFunction.GetLikelihood(solver.DistributionFamily, records, solver.X);
                int k = solver.X.Length;
                fitness_score = -CalcAIC(L, k, n); //negative sign as the lower the AIC, the better the fitted regression model
            }
            else if (criteria == ModelSelectionCriteria.BIC)
            {
                double L_hat = GlmLikelihoodFunction.GetLikelihood(solver.DistributionFamily, records, solver.X);
                int k = solver.X.Length;
                fitness_score = -CalcBIC(L_hat, k, n); //negative sign as the lower the BIC, the better the fitted regression model
            }
            else
            {
                throw new NotImplementedException();
            }

 
            bool improved = true;
            while (improved)
            {
                int eliminatedFeatureId = -1;
                for (int i = 0; i < candidate_features.Count; ++i)
                {
                    List<int> candidate_features_temp = new List<int>();
                    for (int j = 0; j < candidate_features.Count; ++j)
                    {
                        if (i == j) continue;
                        candidate_features_temp.Add(candidate_features[j]);
                    }
                    
                    List<RDataRecord> transformed_data_under_model = RefitModel(candidate_features_temp, solverFactory, records, out solver);

                    double new_fitness_score = -1;
                    if (criteria == ModelSelectionCriteria.AdjustedRSquare)
                    {
                        new_fitness_score = CalcAdjustedRSquare(solver.Statistics.Residuals, outcomes, candidate_features_temp.Count, records.Count);
                    }
                    else if (criteria == ModelSelectionCriteria.AIC)
                    {
                        double L = GlmLikelihoodFunction.GetLikelihood(solver.DistributionFamily, transformed_data_under_model, solver.X);
                        int k = solver.X.Length;
                        new_fitness_score = -CalcAIC(L, k, n); //negative sign as the lower the AIC, the better the fitted regression model
                    }
                    else if (criteria == ModelSelectionCriteria.BIC)
                    {
                        double L_hat = GlmLikelihoodFunction.GetLikelihood(solver.DistributionFamily, transformed_data_under_model, solver.X);
                        int k = solver.X.Length;
                        new_fitness_score = -CalcBIC(L_hat, k, n); //negative sign as the lower the BIC, the better the fitted regression model
                    }

                    if (fitness_score < new_fitness_score)
                    {
                        eliminatedFeatureId = i;
                        fitness_score = new_fitness_score;
                    }
                }

                if (eliminatedFeatureId == -1)
                {
                    improved = false;
                }
                else
                {
                    candidate_features.Remove(eliminatedFeatureId);
                }
            }

            return candidate_features;
        }


        /// <summary>
        /// Calculate the Akaike information criteria for the fitted regression model
        /// The lower the AIC, the better the fitted regression model.
        /// The k term in the AIC penalize the number of parameters added to the regression model
        /// </summary>
        /// <param name="L">The likelihood value of the fitted regression model</param>
        /// <param name="k">The number of fitted parameters (i.e. predictor coefficients in the regression model)</param>
        /// <param name="n">The number of sample data points (i.e. the number of records in the training data)</param>
        /// <returns>The Akaike information criteria</returns>
        public static double CalcAIC(double L, int k, int n)
        {
            double AIC = 2 * k - 2 * System.Math.Log(L);
            double AICc = AIC - 2 * k * (k + 1) / (n - k - 1); //AICc is AIC with a correction for finite sample sizes. 
            return AICc;
        }

        /// <summary>
        /// Calculate the Bayesian information criteria for the fitted regression model
        /// The lower the BIC, the better the fitted regression model
        /// 
        /// When fitting models, it is possible to increase the likelihood by adding parameters, 
        /// but doing so may result in overfitting. Both BIC and AIC resolve this problem by introducing 
        /// a penalty term for the number of parameters in the model; the penalty term is larger in BIC than in AIC.
        /// </summary>
        /// <param name="L_hat">The maximized value of the likelihood function of the model given by the fitted predictor coefficients which minimize the unexplained variability in the regression model)</param>
        /// <param name="k">The number of fitted parameters (i.e. predictor coefficients in the regression model)</param>
        /// <param name="n">The number of sample data points (i.e. the number of records in the training data)</param>
        /// <returns>The Bayesian information criteria</returns>
        public static double CalcBIC(double L_hat, int k, int n)
        {
            double BIC = -2 * System.Math.Log(L_hat) + k * System.Math.Log(n);
            return BIC;
        }

        /// <summary>
        /// Calculate the adjusted R^2 (the explained variablity by the regression model)
        /// The higher the adjusted R^2, the better the fitted regression model
        /// </summary>
        /// <param name="residuals">the residuals {y_i - X_i * beta_hat }_i</param>
        /// <param name="y">The outcomes (i.e. sample values of the response variable)</param>
        /// <param name="k">The number of parameters (i.e. the predictor coefficients in the regression model)</param>
        /// <param name="n">The number of sample data points (i.e., the number records in the training data)</param>
        /// <returns></returns>
        public static double CalcAdjustedRSquare(double[] residuals, double[] y, int k, int n)
        {
            double Var_e = 0;
            for (int i = 0; i < residuals.Length; ++i)
            {
                Var_e = residuals[i] * residuals[i];
            }
            double Var_y = System.Math.Pow(StdDev.GetStdDev(y, Mean.GetMean(y)), 2);
            return 1 - (Var_e / (n - k - 1)) / (Var_y / (n - 1));
        }

        public static Glm FitModel(GlmSolverFactory solverFactory, List<RDataRecord> records)
         {
            DataTransformer<RDataRecord> dt = new DataTransformer<RDataRecord>();
            dt.DoFeaturesScaling(records);

            Glm solver = solverFactory.CreateSolver(records);
            solver.Solve();

            return solver;
         }

         public static List<RDataRecord> RefitModel(List<int> candidate_features, GlmSolverFactory solverFactory, List<RDataRecord> records, out Glm solver)
         {
             List<RDataRecord> records2 = new List<RDataRecord>();

             foreach (RDataRecord rec in records)
             {
                 RDataRecord rec2 = new RDataRecord(candidate_features.Count);
                 
                 for (int d=0; d < candidate_features.Count; ++d)
                 {
                     int featureId = candidate_features[d];
                     rec2[d+1] = rec[featureId];
                 }
                 records2.Add(rec2);
             }

             solver = FitModel(solverFactory, records2);

             return records2;
         }

        /// <summary>
        /// The p-values are P(observed or more extreme coefficients != 0 | true coefficient mean is 0)
        /// </summary>
        /// <param name="CoeffPointEstimates">point estimates of the predictor coefficients</param>
        /// <param name="CoeffSEs">standard errors of the predicator coefficients</param>
        /// <param name="n">number of training records</param>
        /// <param name="one_sided">whether the t distribution is one-sided</param>
        /// <returns>p-values</returns>
         public static double[] CalcPValues(double[] CoeffPointEstimates, double[] CoeffSEs, int n, bool one_sided=false)
         {
             double null_value = 0;
             int k = CoeffPointEstimates.Length;

             double[] pValues = new double[k];
             int df = n - 1;
             for (int i = 0; i < k; ++i)
             {
                 double t = (CoeffPointEstimates[i]-null_value) / CoeffSEs[i];
                 double pValue = (1-StudentT.GetPercentile(System.Math.Abs(t), df)) * (one_sided ? 1 : 2);

                 pValues[i] = pValue;
             }

             return pValues;
         }

         public static int SelectFeatureIndexWithMaxPValue(double[] pValues, out double maxPValue)
         {
             maxPValue = 0;
             int selectedFeatureIndex = -1;
             for (int i = 1; i < pValues.Length; ++i)
             {
                 if (maxPValue < pValues[i])
                 {
                     maxPValue = pValues[i];
                     selectedFeatureIndex = i;
                 }
             }

             return selectedFeatureIndex;
         }
    }
}
