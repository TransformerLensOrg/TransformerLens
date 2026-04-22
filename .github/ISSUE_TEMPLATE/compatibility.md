---
name: Compatibility Report
about: Submit a compatibility report
title: "[Compatibility Report] Model ID"

---

<!--
Use this template to report any issues found with model compatibility. Make sure to create a report per model id, and not per model family. Please include details of any options you used in TransformerLens, and example generations both directly through transformers and TransformerLens. Additionally, please make sure you determine information on historical compatibility with the model in question with TransformerLens.

To determine historic compatibility of your model, first check the performance of the model on the first release of TransformerLens when the model was added. If the incompatibility does not exist on the first version with the model in question, you then need to narrow down the last version where the model performed comparably to transformers, and the first version where it became incompatible. 

The process for finding the last compatible version number is pretty manually. It's a matter of picking a random version, checking the compatibility, and then deciding which version to check next based on the result on the version number in question. If the version you are testing is incompatible, then you want to check earlier releases of TransformerLens. If the version you are testing is compatible, you then want to check newer versions of TransformerLens. This process must be repeated until two consecutive version numbers are found, one where the model was compatible, and the next where the model was incompatible. This process is very tedious, but it will greatly help in the process of fixing the underlying incompatibility.
-->

## Model

REPLACE_WITH_MODEL_ID

- [ ] This model was incompatible when it was introduced to TransformerLens

<!--
Remove the next block if the model in question did not work as expected on the first version of TransformerLens in which it was available.
-->

The model seems to have worked as of REPLACE_WITH_LAST_COMPATIBLE_VERSION_NUMBER. It first started
showing signs of incompatibility in REPLACE_WITH_FIRST_INCOMPATIBLE_VERSION_NUMBER.

### Example of some generations in transformers


### Code used to load the model in TransformerLens


### Example of some generations in TransformerLens
