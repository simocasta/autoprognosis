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
                    step=1
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
                fig.update_layout(
                    font=dict(size=16)  # Adjust the font size as needed
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
            "Outcome Class": predictions.columns,
        }

        # Convert Outcome Class to a list if it is not already
        outcome_classes = list(vals["Outcome Class"])

        if len(outcome_classes) == 2:
            # Map the outcome classes to the desired legend labels for 2 classes
            outcome_classes_mapped = [ 
                {0: "0: No progression", 1: "1: Progression"}.get(int(x), x) 
                for x in outcome_classes
            ]
        else:
            # Map the outcome classes to the desired legend labels for 4 classes
            outcome_classes_mapped = [ 
                {0: "0: No progression", 1: "1: Pain-only progression", 2: "2: Radiographic-only progression", 3: "3: Pain and radiographic progression"}.get(int(x), x) 
                for x in outcome_classes
            ]

        vals["Outcome Class"] = outcome_classes_mapped
        
        # Define custom color sequence
        custom_color_sequence = px.colors.qualitative.Set1
        
        fig = px.bar(
            vals,
            x="Outcome Class",
            y="Probability",
            color="Outcome Class",
            color_discrete_sequence=custom_color_sequence,
            height=450,
            width=600,
        )
        fig.update_layout(
            xaxis_title="Outcome Class",
            yaxis_title="Probability",
            legend_title="Outcome Classes",
            font=dict(size=16)  # Adjust the font size as needed
        )

        st.markdown("<h3 style='color:#000;'>Predictions</h3>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)

    with predictions:
        st.markdown("<h3 style='color:#000;'>Risk estimation</h3>", unsafe_allow_html=True)
        st.markdown(CAUTION_STATEMENT)

        raw_df = pd.DataFrame.from_dict(inputs)
        df = encoders_ctx.encode(raw_df)
        
        # Add a styled button for calculating the risk
        
        if st.button("Show Predictions ✋", help='Click to show predictions'):
            with st.spinner('Processing...'):
                update_predictions(raw_df, df)
                update_interpretation(df)
