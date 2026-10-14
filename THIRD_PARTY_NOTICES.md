# License scope and third-party material

The root MIT license applies to the authors' code in `CODE/` and its repository
documentation. It does not override an existing third-party license or grant
additional rights to the following material.

- **NFv2 data and derived records:** original providers retain their rights.
  The cleaned Parquet releases used here record CC BY-NC-SA 4.0 terms; their
  source/version links are in `DATA/nfv2_processed/raw_data_manifest.json`.
  Raw and processed traffic arrays are not included. Fixed indices, splits,
  and research summaries are provided for scientific verification; they are
  not represented as independently MIT-licensed traffic data.
- **Manuscript and artwork (`paper/`):** copyright remains with the respective
  authors/rightsholders. They are provided for scholarly review and reproducibility;
  the code's MIT license does not license the text or artwork.
- **Elsevier CAS template:** `cas-sc.cls`, `cas-common.sty`,
  `cas-model2-names.bst`, and the accompanying thumbnail assets retain their
  original notices and licenses. The project supplies the subset needed to
  compile this manuscript. The complete original CAS bundle is available from
  [CTAN](https://ctan.org/pkg/els-cas-templates) and is governed by the
  [LaTeX Project Public License](https://www.latex-project.org/lppl/lppl-1-3c/).
- **Dependencies:** Python libraries and the LaTeX distribution are installed
  separately and retain their own licenses.

The implemented learning methods are attributed in the manuscript and code;
research citations do not imply that upstream authors endorse this study.
