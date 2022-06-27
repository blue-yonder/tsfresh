from derive_features_dictionaries import derive_features_dictionaries
from typing import List
from md2pdf.core import md2pdf


def interpret_feature_dynamic(feature_dynamic: str, window_length: int) -> dict:
    assert isinstance(feature_dynamic, str)

    feature_timeseries_mapping, feature_dynamics_mapping = derive_features_dictionaries(
        feature_names=[feature_dynamic]
    )

    return {
        "Full Feature Dynamic Name": feature_dynamic,
        "Input Timeseries": list(feature_timeseries_mapping.keys())[0],
        "Feature Timeseries Calculator": list(feature_timeseries_mapping.values())[0],
        "Window Length": window_length,
        "Feature Dynamic Calculator": list(feature_dynamics_mapping.values())[0],
    }


def dictionary_to_string(dictionary: dict) -> str:
    formatted_output = ""
    for key, value in dictionary.items():
        formatted_output += f"**{key}** : ```{value}```<br>"
    return formatted_output


def gen_pdf_for_feature_dynamics(
    feature_dynamics_names: List[str],
    window_length: int,
    output_path: str = "feature_dynamics_interpretation",
) -> None:
    """ """
    feature_dynamics_summary = "<br/><br/><br/>".join(
        [
            dictionary_to_string(
                interpret_feature_dynamic(
                    feature_dynamic=feature_dynamics_name,
                    window_length=window_length,
                )
            )
            for feature_dynamics_name in feature_dynamics_names
        ]
    )

    title = "# Feature Dynamics Summary"
    linebreak = "---"
    context = "**Read more at:**"
    link1 = "* [How to interpret feature dynamics](www.google.com)"
    link2 = "* [List of feature calculators](https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html)"

    with open("feature_dynamics_interpretation.md", "w") as f:
        f.write(
            f"{title}\n\n{linebreak}\n\n{context}\n\n{link1}\n\n{link2}\n\n{linebreak}\n\n{feature_dynamics_summary}"
        )

    md2pdf(
        pdf_file_path=f"{output_path}.pdf",
        md_file_path=f"{output_path}.md",
    )
