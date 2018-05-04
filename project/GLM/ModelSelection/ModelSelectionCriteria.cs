using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace GlmSharp.ModelSelection
{

    public enum ModelSelectionCriteria
    {
        pValue, //p-value for the predictor coefficient in the null hypothesis
        AdjustedRSquare, //Adjusted explained variablity in the fitted regression model 
        AIC, //Akaikane Information Criteria
        BIC //Bayesian Information Criteria
    }
}
