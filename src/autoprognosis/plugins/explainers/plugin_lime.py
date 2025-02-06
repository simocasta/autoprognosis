# stdlib
import copy
from typing import Any, List, Optional

# third party
import numpy as np
import pandas as pd

# autoprognosis absolute
from autoprognosis.plugins.explainers.base import ExplainerPlugin
from autoprognosis.utils.pip import install

for retry in range(2):
    try:
        # third party
        import lime
        import lime.lime_tabular

        break
    except ImportError:
        depends = ["lime"]
        install(depends)


class LimePlugin(ExplainerPlugin):
    """
    Interpretability plugin based on LIME.

    Args:
        estimator: model. The model to explain.
        X: dataframe. Training set
        y: dataframe. Training labels
        task_type: str. classification of risk_estimation
        prefit: bool. If true, the estimator won't be trained.
        n_epoch: int. training epochs
        time_to_event: dataframe. Used for risk estimation tasks.
        eval_times: list. Used for risk estimation tasks.

    Example:
        >>> import pandas as pd
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>>from autoprognosis.plugins.explainers import Explainers
        >>> from autoprognosis.plugins.prediction.classifiers import Classifiers
        >>>
        >>> X, y = load_iris(return_X_y=True)
        >>>
        >>> X = pd.DataFrame(X)
        >>> y = pd.Series(y)
        >>>
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        >>> model = Classifiers().get("logistic_regression")
        >>>
        >>> explainer = Explainers().get(
        >>>     "lime",
        >>>     model,
        >>>     X_train,
        >>>     y_train,
        >>>     task_type="classification",
        >>> )
        >>>
        >>> explainer.explain(X_test)
    """

    def __init__(
        self,
        estimator: Any,
        X: pd.DataFrame,
        y: pd.DataFrame,
        feature_names: Optional[List] = None,
        task_type: str = "classification",
        prefit: bool = False,
        n_epoch: int = 10000,
        # Risk estimation
        time_to_event: Optional[pd.DataFrame] = None,  # for survival analysis
        eval_times: Optional[List] = None,  # for survival analysis
        random_state: int = 0,
        categorical_features: Optional[List[int]] = None,  # allow user override
        **kwargs: Any,
    ) -> None:
        if task_type not in ["classification", "risk_estimation"]:
            raise RuntimeError("invalid task type")
    
        self.task_type = task_type
        self.feature_names = list(
            feature_names if feature_names is not None else pd.DataFrame(X).columns
        )
        super().__init__(self.feature_names)
    
        model = copy.deepcopy(estimator)
        if task_type == "classification":
            if not prefit:
                model.fit(X, y)
    
            def model_fn(X_in: pd.DataFrame) -> pd.DataFrame:
                X_in = pd.DataFrame(X_in, columns=self.feature_names)
                return np.asarray(model.predict_proba(X_in)).astype(float)
    
            self.predict_fn = model_fn
        elif task_type == "risk_estimation":
            if time_to_event is None or eval_times is None:
                raise RuntimeError("invalid input for risk estimation interpretability")
    
            if not prefit:
                model.fit(X, time_to_event, y)
    
            def model_fn(X_in: pd.DataFrame) -> pd.DataFrame:
                X_in = pd.DataFrame(X_in, columns=self.feature_names)
                return np.asarray(model.predict(X_in, eval_times)).astype(float)
    
            self.predict_fn = model_fn
    
        # Auto-detect categorical features if not provided
        if categorical_features is None:
            cat_feats = []
            X_df = pd.DataFrame(X, columns=self.feature_names)
            for idx, col in enumerate(X_df.columns):
                # Check if the column has an object or categorical dtype,
                # or if it has very few unique values relative to the dataset size.
                if pd.api.types.is_object_dtype(X_df[col]) or pd.api.types.is_categorical_dtype(X_df[col]):
                    cat_feats.append(idx)
                elif X_df[col].nunique() <= 10:  # threshold can be adjusted
                    cat_feats.append(idx)
            categorical_features = cat_feats
    
        # If task is classification, we assume mode "classification" for LIME
        if task_type == "classification":
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                np.asarray(X),
                feature_names=self.feature_names,
                categorical_features=categorical_features,
                discretize_continuous=False  # disable discretisation for categorical features
            )
        else:
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                np.asarray(X),
                feature_names=self.feature_names,
                categorical_features=categorical_features,
                mode="regression"
            )


    def explain(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X, columns=self.feature_names)
        X = np.asarray(X)
        results = []

        for v in X:
            expl = self.explainer.explain_instance(
                v,
                self.predict_fn,  # labels=self.feature_names,# top_labels=self.feature_names
            )
            importance = expl.as_list(label=1)

            vals = [x[1] for x in importance]
            cols = [x[0] for x in importance]
            results.append(vals)

        return pd.DataFrame(results, columns=cols)

    @staticmethod
    def name() -> str:
        return "lime"

    @staticmethod
    def pretty_name() -> str:
        return "LIME"


plugin = LimePlugin
