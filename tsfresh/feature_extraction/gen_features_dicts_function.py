from tsfresh.feature_extraction.settings import from_columns
from typing import List
import shutil
from md2pdf.core import md2pdf


def derive_features_dictionaries(feature_names: List[str]):
    """
    Derives and writes out two feature dictionaries which can be used with the feature dynamics framework.

    Return the dictionaries as a single object, and a flag specifying what type of dictionary... i.e. if it is columns --> feature dict

        params:
            feature_names (list of str): the relevant feature names in the form of <ts_kind>||<feature_time_series>__<feature_dynamic>

        returns:
            f_mapping (dict):
            f_on_f_mapping (dict):
    """

    # type check might not be neccessary
    # assert feature_names and all(isinstance(feature_dynamic, str) for feature_dynamic in feature_names)

    replacement_token = "||"  # set this as the standard as per the docstring...

    f_on_f_mapping = from_columns(feature_names)
    f_mapping = from_columns(
        [str(x).replace(replacement_token, "__") for x in [*f_on_f_mapping]]
    )

    return f_mapping, f_on_f_mapping


def interpret_feature_dynamic(feature_dynamic: str, sub_feature_split: int):
    assert isinstance(feature_dynamic, str)

    f_mapping, f_on_f_mapping = derive_features_dictionaries(
        feature_names=[feature_dynamic]
    )

    return {
        "Full Feature Dynamic Name": feature_dynamic,
        "Input time series": list(f_mapping.keys())[0],
        "Feature time series": list(f_on_f_mapping.keys())[0],
        "Window Size": sub_feature_split,
        "Feature Dynamic": list(f_on_f_mapping.values())[0],
    }


def format_output_for_a_summary(summary):
    formatted_output = ""
    for key, value in summary.items():
        formatted_output += f"**{key}** : `{value}`<br>"
    return formatted_output


def gen_pdf_for_feature_dynamics(
    feature_dynamics_names: List[str], sub_feature_split: int
) -> None:
    """ """
    feature_dynamics_summary = "\n\n\n".join(
        [
            format_output_for_a_summary(
                interpret_feature_dynamic(
                    feature_dynamic=feature_dynamics_name,
                    sub_feature_split=sub_feature_split,
                )
            )
            for feature_dynamics_name in feature_dynamics_names
        ]
    )

    with open("feature_dynamics_interpretation.md", "w") as f:
        f.write("# Feature Dynamics Summary\n\n" + feature_dynamics_summary)

    md2pdf(
        pdf_file_path="feature_dynamics_interpretation.pdf",
        md_file_path="feature_dynamics_interpretation.md",
    )
