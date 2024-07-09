# stdlib
from typing import Any, Dict, List

# third party
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# autoprognosis absolute
import autoprognosis.logger as log
from autoprognosis.plugins.explainers import Explainers
from autoprognosis.utils.data_encoder import EncodersCallbacks
from autoprognosis.utils.pip import install

# Import necessary packages for retry mechanism
for retry in range(2):
    try:
        import plotly.express as px
        import streamlit as st
        break
    except ImportError:
        depends = ["streamlit", "plotly"]
        install(depends)


def classification_dashboard(
    title: str,
    banner_title: str,
    models: Dict,
    column_types: List,
    encoders_ctx: EncodersCallbacks,
    menu_components: List,
    plot_alternatives: Dict,
) -> Any:
    """
    Streamlit helper for rendering the dashboard, using serialized models and menu components.

    Args:
        title:
            Page title
        banner_title:
            The title used in the banner.
        models:
            The models used for evaluation and plots.
        column_types: List
            List of the dataset features and their distribution.
        encoders_ctx: EncodersCallbacks,
            List of encoders/decoders for the menu values < - > model input values.
        menu_components: List
            Type of menu item for each feature: checkbox, dropdown etc.
        plot_alternatives: list
            List of features where to plot alternative values. Example: if treatment == 0, it will plot alternative treatment == 1 as well, as a comparison.
    """

    hide_footer_style = """
        <style>
        .reportview-container .main footer {visibility: hidden;}
        </style>
    """
    st.markdown(hide_footer_style, unsafe_allow_html=True)
    st.markdown(
        """ <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    CAUTION_STATEMENT = "This tool predicts your most likely outcomes based on current knowledge and data, but will never provide a 100% accurate prediction for any individual. We recommend that you discuss the results with your own specialist in a more personalised context."

    menu, predictions = st.columns([1, 4])

    inputs = {}
    columns = []
    with menu:
        st.markdown("<h3 style='color:#000;'>Patient info</h3>", unsafe_allow_html=True)
        
        for name, item in menu_components:
            columns.append(name)
            if item.type == "checkbox":
                obj = st.checkbox(
                    label=item.name,
                )
                inputs[name] = [obj]
            if item.type == "dropdown":
                if name in ["KL grade", "Medial JSN", "Lateral JSN"]:
                    sorted_val_range = [0, 1, 2, 3]
                else:
                    # Sort the val_range attribute for the dropdown options
                    def try_float(value):
                        try:
                            return float(value)
                        except ValueError:
                            return None

                    def is_numeric(value):
                        return try_float(value) is not None

                    numerical_values = sorted([x for x in item.val_range if is_numeric(str(x))], key=try_float)
                    non_numerical_values = sorted([x for x in item.val_range if not is_numeric(str(x))])

                    sorted_val_range = numerical_values + non_numerical_values

                obj = st.selectbox(
                    label=item.name,
                    options=[val for val in sorted_val_range],
                )
                inputs[name] = [obj]
            elif item.type == "slider_integer":
                max_value_adjusted = ((item.max // 5) + 1) * 5
                min_value = item.min if name != "Age" else 45
                max_value = item.max if name != "Age" else max_value_adjusted
                obj = st.slider(
                    item.name,
                    min_value=min_value,
                    value=item.min,
                    max_value=max_value,
                    step=5
                )
                inputs[name] = [obj]
            elif item.type == "slider_float":
                max_value_adjusted = ((int(item.max) // 5) + 1) * 5
                min_value = 0.0
                step_value = 0.1
                if name == "WOMAC Pain Score":
                    max_value = 20.0
                elif name == "WOMAC Disability Score":
                    max_value = 68.0
                else:
                    max_value = float(max_value_adjusted)
                obj = st.slider(
                    item.name,
                    min_value=min_value,
                    value=float(item.min),
                    max_value=max_value,
                    step=step_value,
                )
                inputs[name] = [obj]

    def update_interpretation(df: pd.DataFrame) -> px.imshow:
        for reason in models:
            if not hasattr(models[reason], "explain"):
                log.error(f"Ignoring model for XAI {reason}")
                continue
            try:
                raw_interpretation = models[reason].explain(df)
                if not isinstance(raw_interpretation, dict):
                    raise ValueError("raw_interpretation must be a dict")
            except BaseException:
                log.error(f"Failed to get reason {reason}")
                continue

            for src in raw_interpretation:
                pretty_name = Explainers().get_type(src).pretty_name()
                src_interpretation = raw_interpretation[src]

                if src_interpretation.shape != (1, len(df.columns)):
                    log.error(
                        f"Interpretation source provided an invalid output {src_interpretation.shape}. expected {(1, len(df.columns))}"
                    )
                    continue

                src_interpretation = np.asarray(src_interpretation)

                interpretation_df = pd.DataFrame(
                    src_interpretation[0, :].reshape(1, -1),
                    columns=df.columns,
                    index=df.index.copy(),
                )
                interpretation_df = encoders_ctx.numeric_decode(interpretation_df)

                fig = px.imshow(
                    interpretation_df,
                    labels=dict(x="Feature", y="Source", color="Feature importance"),
                    color_continuous_scale="Blues",
                    height=300,
                )
                st.markdown(
                    f"<h3 style='color:#000;'>Feature importance for the '{reason}' risk plot using {pretty_name}</h3>", 
                    unsafe_allow_html=True
                )
                st.plotly_chart(fig, use_container_width=True)

    def update_predictions(raw_df: pd.DataFrame, df: pd.DataFrame) -> px.imshow:
        for reason in models:
            predictions = models[reason].predict_proba(df)
            break

        vals = {
            "Probability": predictions.values.squeeze(),
            "Category": predictions.columns,
        }
        fig = px.bar(
            vals,
            x="Category",
            y="Probability",
            color="Category",
            color_continuous_scale="RdBu",
            height=600,
            width=600,
        )
        fig.update_layout(
            xaxis_title="Category",
            yaxis_title="Probability",
            legend_title="Categories",
        )

        st.markdown("<h3 style='color:#000;'>Predictions</h3>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)

    with predictions:
        st.markdown("<h3 style='color:#000;'>Risk estimation</h3>", unsafe_allow_html=True)
        st.markdown(CAUTION_STATEMENT)

        raw_df = pd.DataFrame.from_dict(inputs)
        df = encoders_ctx.encode(raw_df)
        
        # Add a styled button for calculating the risk
        button_style = """
        <style>
        div.stButton button {
            background-color: #1f77b4 !important;
            color: white !important;
            padding: 20px 40px !important;
            font-size: 36px !important; /* Increase the text size */
            border-radius: 12px !important;
            border: none !important;
            cursor: pointer !important;
            box-shadow: 0 6px #999 !important;
        }
        div.stButton button:hover {
            background-color: #0056b3 !important;
        }
        div.stButton button:active {
            background-color: #0056b3 !important;
            box-shadow: 0 4px #666 !important;
            transform: translateY(2px) !important;
        }
        </style>
        """
        # st.markdown(button_style, unsafe_allow_html=True)
        if st.button("Show Predictions ✋"):
            update_predictions(raw_df, df)
            update_interpretation(df)
