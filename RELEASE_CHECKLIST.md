# GitHub + Zenodo release checklist

Before the first public release:

1. **Author names** — verify the canonical full bibliographic form of both authors in `CITATION.cff` and `.zenodo.json`.
2. **Affiliations** — keep North-West University first if that is the desired manuscript ordering.
3. **License** — choose the public license before release. A software-specific license such as BSD-3-Clause or MIT is usually appropriate for code; if figures/data need a separate license, document that explicitly.
4. **Repository visibility** — keep GitHub private until the JCAM submission is complete if desired.
5. **Run smoke tests** — build the Docker image and execute at least one small manufactured case plus both validation scripts.
6. **Check archived data** — ensure CSV files and figure captions agree with the submitted manuscript.
7. **Create Git tag** — recommended first tag: `v1.0.0`.
8. **Connect GitHub to Zenodo** — enable the repository in Zenodo/GitHub integration, then create the GitHub release.
9. **Zenodo DOI** — after deposition, add the concept DOI/release DOI badge and citation to `README.md` and `CITATION.cff`.
10. **JCAM submission** — provide the public repository/DOI in the Data Availability or Code Availability statement if the journal submission form requests it.

## Recommended repository name

`pressure-robust-cbf-bdm-sip-bdf2`

## Recommended release title

`v1.0.0 - Reproducibility package for the JCAM submission`
