<!--
When opening your PR, please make sure to only request a merge to `main` when you have found a bug in the currently released version of TransformerLens. All other PRs should go to `dev` in order to keep the docs in sync with the currently released version.

Please also make sure the branch you are attempting to merge from is not named `main`, or `dev`. Branches with these names from a different remote cause conflicting name issues when we periodically attempt to bring your PR up to date with the current stable TransformerLens source.

If your PR is primarily affecting docs, make sure has the string "docs" in its name. Building docs is disabled by default to avoid CI time, but the job has been configured to run whenever a branch with the word "docs" in it is being merged.
-->
# Description

Please include a summary of the change and which issue is fixed. Please also include relevant motivation and context. List any dependencies that are required for this change.

Fixes # (issue)

## Type of change

Please delete options that are not relevant.

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] This change requires a documentation update

### Screenshots
Please attach before and after screenshots of the change if applicable.

<!--
Example:

| Before | After |
| ------ | ----- |
| _gif/png before_ | _gif/png after_ |


To upload images to a PR -- simply drag and drop an image while in edit mode and it should upload the image directly. You can then paste that source into the above before/after sections.
-->

# Checklist:

- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have not rewritten tests relating to key interfaces which would affect backward compatibility

<!--
As you go through the checklist above, you can mark something as done by putting an x character in it

For example,
- [x] I have done this task
- [ ] I have not done this task
-->