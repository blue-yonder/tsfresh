def gen_pdf_for_feature_dynamics(
    feature_dynamics_names: List[str], window_length: int
) -> None:
    """ """
    feature_dynamics_summary = "\n\n\n".join(
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
        pdf_file_path="feature_dynamics_interpretation.pdf",
        md_file_path="feature_dynamics_interpretation.md",
    )
