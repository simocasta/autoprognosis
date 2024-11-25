
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

    CAUTION_STATEMENT = "This application is a prototype for demonstrative purposes only. It is NOT intended for use on any individual, including in any clinical or medical setting."

    menu, predictions = st.columns([1, 4])

    inputs = {}
    columns = []
    missing_count = 0

    with menu:
        st.markdown("<h3 style='color:#000;'>Patient info</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 2])
        with col2:
            st.markdown("<h5 style='font-size:14px;'>Missing?</h5>", unsafe_allow_html=True)
        
        for name, item in menu_components:
            columns.append(name)
        
            col1, col2 = st.columns([4, 2])
        
            with col2:
                missing_checkbox = st.checkbox("", key=f"{name}_missing")
        
            with col1:
                if missing_checkbox:
                    obj = np.nan
                    missing_count += 1
                    st.markdown(f"<p style='font-family: Arial, sans-serif; font-size: 14px;'>{name}</p>", unsafe_allow_html=True)  # Keep the variable name displayed
                else:
                    if item.type == "checkbox":
                        obj = st.checkbox(label=name)
                        
                    elif item.type == "dropdown":
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
                            label=name,
                            options=[val for val in sorted_val_range],
                        )
                           
                    elif item.type == "slider_integer":
                        min_value = item.min
                        max_value = item.max
                        obj = st.slider(
                            name,
                            min_value=min_value,
                            value=item.min,
                            max_value=max_value,
                            step=1
                        )
        
                    elif item.type == "slider_float":
                        max_value_adjusted = int(item.max)
                        min_value = 0.0
                        step_value = 0.1
                        max_value = float(max_value_adjusted)
                        obj = st.slider(
                            name,
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
                {0: "0: Normal", 1: "1: Abnormal"}.get(int(x), x) 
                for x in outcome_classes
            ]
        else:
            # Map the outcome classes to the desired legend labels for 4 classes
            outcome_classes_mapped = [ 
                {0: "0: Normal", 1: "1: Hernia", 2: "2: Spondylolisthesis"}.get(int(x), x) 
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
                if missing_count > 3:
                    st.error("Too many missing values")
                else:
                    update_predictions(raw_df, df)
                    update_interpretation(df)

