# Data and third-party material policy

The repository licence covers the original code and documentation only. It
does not relicense the public datasets, pretrained models, or third-party
figures used by BanglaCyberBench.

## Source datasets

The benchmark consolidates four public sources. Users must obtain each source
from its authoritative landing page and follow the licence, attribution,
privacy, and redistribution conditions stated there.

| Source | Public reference | Repository policy |
|---|---|---|
| Facebook-44K | Ahmed et al.; Mendeley Data DOI `10.17632/9xjx8twk8p.1` | Do not redistribute raw text unless the source terms permit it. |
| BanTH | Haider et al., Findings NAACL 2025; ACL DOI `10.18653/v1/2025.findings-naacl.403` | Obtain from the official release and preserve its attribution and terms. |
| Bangla multilabel dataset | Sunny et al.; Mendeley Data DOI `10.17632/sz5558wrd4.3` | Obtain from Mendeley Data and follow its licence. |
| BD-SHS | Romim et al., LREC 2022; ACL URL `https://aclanthology.org/2022.lrec-1.552/` | Follow the paper/repository terms; raw text is not automatically covered by MIT. |

The repository may distribute preprocessing code, label mappings, UID-level
manifests, checksums, prediction arrays, and split metadata. A researcher who
cannot legally download raw text should not attempt to reconstruct or publish
it from derived artefacts.

## Pretrained models

BanglishBERT, BanglaBERT, MuRIL, XLM-R, and their tokenizers are third-party
works. Users must comply with the licence and model-card terms of each model
provider. Model weights are not relicensed by this repository.

## Sensitive content

The data contains abusive, sexual, religious, and threatening language. It is
provided for research on abuse detection and content moderation. Users should
not use it to harass, profile, target, or generate abuse against individuals or
groups.
