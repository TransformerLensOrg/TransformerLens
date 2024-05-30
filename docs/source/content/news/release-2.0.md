# TransformerLens 2.0

I am very happy to announce TransformerLens now has a 2.0 release! If you have been using recent versions of TransformerLens, then the good news is that not much has changed at all. The primary motivation behind this jump is to transition the project to strictly following semantic versioning as described [here](https://semver.org/). At the last minute we did also remove the recently added HookedSAE, so if you had been using that, I would direct you to Joseph Bloom’s [SAELens](http://github.com/jbloomAus/SAELens). Bundled with this major version change are also a handful of internal modifications that only affect contributors.

## First, an introduction

My name is Bryce Meyer. I am a software engineer, with just a little under 15 years of professional experience with a wide range of expertise from embedded computing to API design. Within the last couple years I have gotten more and more involved in ML, and especially AI safety within the last nine months. At the end of March, I was chatting a bit with Joseph Bloom, and he asked me if I might be interested in taking on the role as primary maintainer of TransformerLens. I have been doing so on a part time basis since the beginning of April, and so far I am pretty happy with the progress that has been made.

In this first month, with the help of the many kind contributors to TransformerLens we have managed to address every single pull request, many of which had been awaiting reply for quite some time. In total around 30 pull requests have been merged with contributions from around 20 people. Within those PRs a number of features were added, including, but not limited to support for Mixtral, LLaMA 3, 4-bit Quantized LLaMA, and HookedSAETransformer, a brand new class to splice sparse autoencoders into HookedTransformer.

I have two primary immediate goals for my time as primary maintainer of this project. The first is to position TransformerLens in a way where it is approachable to as many people as possible, while also remaining powerful for people who are pushing the limits of the field. My second goal is to find ways to make the code base for this project easier to manage for the future, as the development and availability of LLMs continues to accelerate.

I feel that this project has a massive amount of momentum at the moment, and I am hoping to carry that over into the future with new features. I have a software engineering background, not research, and I know I need to talk a lot with users to ensure the library is meeting their needs. I have personally spoken with around a dozen people in the community on their experience with TransformerLens, and what they may want to see happen in the future. If you have not spoken with me, but you would like to, please [make an appointment](https://calendly.com/bryce-c7e/30min). I am curious to hear from anyone who is using this tool, from absolute beginner to complete experts.

## Adopting Semantic Versioning

In the last month, a lot has changed with TransformerLens. Not only have a lot of changes been made to the code, but ideas of how the project will be managed are also evolving. The biggest change in the management of the project is the previously mentioned adoption of Semantic Versioning. Previously, the project had not been officially managed under Semantic Versioning, and there were instances where compatibility was not maintained through the 1.x branch. Going forward, API changes will be managed strictly, and in a way that maintains compatibility through major versions. If you are starting a project today using TransformerLens 2.0, then you can be rest assured that you will be able to upgrade your code all the way through the 2.x branch without worrying about needing to make changes to your code. For full details of how Semantic Versioning will affect TransformerLens, please see the appendix.

## Deprecations

There are right now two deprecations in the code base. The parameter `move_model` in `ActivationCache.to`, and the function `cache_all` in `hook_points`. To keep things simple for the change to semantic versioning, they will be remaining. However, if you are using them, then make sure to adapt your code right away. They will be removed in 3.0. Along with that, anything new that is marked deprecated in 2.x will also be removed when the next major version comes around.

Whenever something new does become deprecated, it will also be prominently noted in the release notes to make sure these sorts of things do not slip by. In my previously mentioned scenario where a key was renamed, in the future a situation like this will be handled by changing the code to the new key, but then persisting the deprecated old key pointing at the new key. This will allow anyone relying on that key to continue to do so without interruption.

However, I would still encourage everyone to periodically check release notes to make sure they are keeping an eye out for when things become deprecated. I don’t imagine it will happen often, but it will happen. Keeping an eye on it will save a lot of trouble in the future when moving from one major release to the next.

## Roadmap

There are three primary timeframes that I am approaching the current development plans of TransformerLens.

### Immediate - within the next month

At the moment, TransformerLens is in a state where all pull requests are being addressed quickly, but the issue tracker is still full of items that have not been addressed. The first thing to be done now is to go through all issues, categorize them, and address anything that is easy to address. Once that is done, both issues and pull requests will be up to date, and should remain that way going forward.

### Mid-term - within the next 3 months

Please note that the below is a draft roadmap. We are very happy to change our prioritization if user feedback surfaces other issues.

#### Performance

One of the first new items to be improved in TransformerLens is general performance. This will be achieved in a couple ways. The first involves diagnosing various areas in the code where memory leaks may be occurring. It seems like there are a number of places where references are not properly being released, thus causing garbage collection to not run correctly. If we can identify these places, and find proper ways to run garbage collection, we should be able to improve overall performance a lot, especially when dealing with larger models.

The second performance task will be exploring ways to improve the ability to batch processes. I have already had code shared with myself that seems to work well at batching just about anything together in a very general way. There is also a separate volunteer working on going through said code and finding a good implementation to add to TransformerLens.

#### Streamlining Adding New Models

Improving the model submission process is another item to be addressed in the near future. It may be a good use of time to have some open discussions on this, but I do think this needs to be improved. In my discussions these last few weeks, I have found two primary problems involving models. The first is a general confusion among a good number of people on how they would go about adding a model to TransformerLens. The second problem is ensuring that the logits calculated within TransformerLens match HuggingFace. The goal is to solve both of these problems by systematizing the submission process (e.g. with good tutorials), and requiring that all new models submitted to the project have calculated logits to show that the models they are submitting match HuggingFace. This requirement at the moment would be quite a bit to ask for contributors. In order to alleviate that problem, we will build a little tool that will automatically calculate these logits, and spit out a table of said logits to be submitted with the PR. This would also give us the ability to snapshot said logits, store them in the repo, and periodically regenerate them to make sure cumulative changes to the code base have not affected these values.

### Long-term - within the next year

#### Model Testing

Finding a way to create more robust tests with models integrated is a pretty big item that has been discussed. This is already implemented for the smaller models in each family, but is hard for model families like LLaMA where even the smallest is 7B. A lot of thoughts have been thrown around on this topic, but our best guess for a reasonable solution is to create an untrained small version of the model on HuggingFace (eg randomly initialized weights) and to verify that we can load that in. The resulting tests would not be accurate in the sense that using the full model would be, but it would allow testing for consistency on a smaller sample size of the larger model, and thus allow for the ability to test code against those more bite sized models. If we can find a successful way to go about handling this, then this could turn into a resource available for a lot of other projects to allow people to write efficient integration tests.

#### Model Integration

Making it easier for TransformerLens to handle a large range of models is another long term project that is at the moment a very hard problem to solve. Doing so, however, will be key to ensuring that the library is more future-proof. There have been many ideas put out of how to solve this, and this seems to be a topic that a lot of people have very strong opinions on. Most likely, there will need to be a handful of roundtable discussions on this to find the best solution. 

One of the ideas is to have a generalized wrapper that can take in a model from HuggingFace. Another idea is to create a way to allow TransformerLens to have plugins, so addition of models can be handled outside of the main project, and people can publish compatibility with new models themselves without having to put them into the main project. Finally, there is an idea to keep the submission within TransformerLens as is, but to overhaul the way they are configured so that code can be shared more across models, and something like configuration composition can then be utilized to allow for common cases to be tested in isolation, and thus more rapidly allowing new settings to be accepted, and for people to attempt to configure models themselves without having to have the configuration itself in the repo. 

It is very likely that none of these current ideas will end up being the solution, but we do need to come up with a solution. In my discussions with the community, one of the most common pain points was not having model compatibility. Up to now, it has been managed relatively well, but we still end up taking some time to accept new models, and there are a lot of models that TransformerLens does not support. This problem is only going to grow exponentially as the amount of available models grows as the whole field explodes.

## Contributors

This next section is only relevant to contributors, so if anyone is reading this who is only using TransformerLens as a tool, then you can skip this section. 

### New Dev Branches

There have been two new branches setup. One with the last release of 1.x, and another that will act as the current active development branch. The vast majority of pull requests should be made to the new dev branch. The reason for doing this is due to a potential mismatch between docs and the last release. Previously, all pull requests were put into the main branch, which caused the docs to be generated This meant that there were quite a few instances where the docs referenced features and functions that had not yet been released to people installing via pip. 

From now on, dev will represent the most up to date version of the code with the vast majority of pull requests going to dev. The old main branch will only be updated when a release is ready to be sent out. Releases will be sent out when there are enough changes in dev to justify a release, and when bugs are found in the current release with PRs fixing those bugs going directly into main with an immediate release following.

### Integration Tests

The existing unit tests have been split out into two main groups. Those groups are now called integration tests, and, once again, unit tests. A lot of the existing unit tests were testing various pieces of the code base, and outside resources working together. Traditionally, unit tests should test everything in absolute isolation. That means that if the code is working with other parts of the code base, or outside resources, those pieces should be mocked out, and spied on to absolutely control all input and output of the functions, including side effects of said functions. The goal of this is to be able to be absolutely certain that the logic in the tested functions is being tested in absolute isolation, so that bugs can be entirely ruled out within that functions logic.

A lot of the tests that would be categorized as integration, are still incredibly useful, but they are useful in a different way. To make both of them more useful in general, it makes sense to separate them. Unit tests are the first place to look when fixing a bug in a code base, but if they are dealing with outside resources, you cannot be absolutely certain that the bug does not originate from the outside resource. If the unit tests all test code in isolation, then if all of your unit tests pass, but your integration tests do not, then you can immediately rule out a whole bunch of places where bugs may be possible, and start looking in different places for bugs. Being able to separate the two test styles is going to make everyones lives a lot easier when it comes to maintaining the project.

### Test Coverage
 
The CI has recently been updated to publish a test coverage report. The overall coverage of the project should be improved. If anyone would like to begin improving the coverage, that could be a great way to start getting involved. There are quite a few parts of the code base that have no coverage. Reviewing that report, and finding a place to write a good, meaningful test is a great way to get started, and it should be easier than ever to do so. I have a handful of unit tests that I would like to have written that will improve coverage substantially, so if anyone would like to volunteer to take on some of those, let me know and I will be happy to point you in the right direction. 

### Components Refactor

This is the biggest change to the actual code base in this shift to 2.0. The whole components.py file has been removed in favor of a full module with individual files for each class that was previously in that file. The file was approaching 3000 lines with 18 distinct classes within it. There was no order to it either, since a lot of components were interdependent on each other, and it thus ended up being ordered from least dependent to most dependent. Now, everything has its own file, and every class is really easy to find in case of needing to reference anything. The refactor has been done in a way where no code needed to be changed anywhere else in the project, or outside the project. If you have been doing something like `import MLP from transformer_lens.components`, that will work, and continue to work exactly the same.

## Conclusion

Thank you very much for taking the time to read through this. I am very excited to be able to take on maintaining this project, and I am hoping to be able to work on this full time for at least the next year. I am not a researcher, and I am approaching all of this from a software engineering standpoint. This does give me a bit of an outsider's perspective on this in comparison to a lot of the people that have put in so much work to make this tool worth using. 

I really think this tool is incredibly important, and it is enabling research that is going to have a huge impact on the world. My hope is that I can bring my expertise in software engineering to this project, so that the researchers using this tool can be more efficient, and so they can get to the more important tasks they need to complete. I will do my best to make that a reality.

## Appendix

### Semantic Versioning

One of the main goals of semantic versioning (semver) is to communicate to people using the software if a new version is going to be completely compatible with an older version, or if they may need to check a change log to see if a feature they are using has changed. In the case of something like TransformerLens, the full API itself is what we base our version changes on. That includes classes, functions, parameters to functions, and data returned from said functions, including all exposed object properties or keys. When anything is added, then the minor version number gets a bump. When the API remains the same, but a bug has been fixed, the patch number gets a bump. Finally, when anything is removed whatsoever, then that requires a major version change. 

With TransformerLens, the reason why it is necessary in this transition to bump to 2.0 is simply due to the fact that things from 1.0 have been removed along the 1.x branch, thus breaking the exposed API. In most cases, the breaking changes were simple things like exposed keys being renamed. I discovered a handful of these cases within the last month while bringing demos up from earlier versions of TransformerLens into more recent versions of 1.x. 

I do not know exactly what the full extent of these changes were, and doing an exploration of this is probably not a great use of time. Regardless, the point stands that if you have a project using TransformerLens 1.0, you cannot reliably upgrade to 1.17 without possibly needing to change your code due to these exposed changes.The easiest thing to do in this situation while adopting semver is to simply bump the major version, and start fresh.

Going forward, code that you write using TransformerLens will work, and will consume the same minimal API for all minor and patch releases in the 2.x branch.
