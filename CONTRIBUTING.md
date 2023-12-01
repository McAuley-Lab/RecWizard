<!---
Copyright 2023 Julian McAuley Lab UCSD. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# Contribute to <img src="server/recwizard.png" alt="recwizard logo" width="25"/> RecWizard

Everyone is welcome to contribute, and we value everybody's contribution. Code contributions are not the only way to help the community. Answering questions, helping others, and improving the documentation are also immensely valuable.

It also helps us if you spread the word! Reference the library in blog posts
about the awesome projects it made possible, shout out on Twitter every time it has helped you, or simply ⭐️ the repository to say thank you.

However you choose to contribute, please be mindful and respect our
[code of conduct](https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md).

**This guide was heavily inspired by the awesome [transformers guide to contributing](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md).**

## Ways to contribute

There are several ways you can contribute to <img src="server/recwizard.png" alt="recwizard logo" width="15"/> Recwizard:

* Fix outstanding issues with the existing code.
* Submit issues related to bugs or desired new features.
* Implement new models.
* Contribute to the examples or to the documentation.

> All contributions are equally valuable to the community. 🥰

## Fixing outstanding issues

If you notice an issue with the existing code and have a fix in mind, feel free to [start contributing](https://github.com/McAuley-Lab/RecWizard/blob/main/CONTRIBUTING.md/#create-a-pull-request) and open a Pull Request!

## Submitting a bug-related issue or feature request

Do your best to follow these guidelines when submitting a bug-related issue or a feature request. It will make it easier for us to come back to you quickly and with good feedback.

### Did you find a bug?

The Recwizard library will aspire to be robust and reliable thanks to users who report the problems they encounter.

Before you report an issue, we would really appreciate it if you could **make sure the bug was not
already reported** (use the search bar on GitHub under Issues). Your issue should also be related to bugs in the library itself, and not your code. 

Once you've confirmed the bug hasn't already been reported, please include the following information in your issue so we can quickly resolve it:

* Your **OS type and version** and **Python**, **PyTorch** and
  **TensorFlow** versions when applicable.
* A short, self-contained, code snippet that allows us to reproduce the bug in
  less than 30s.
* The *full* traceback if an exception is raised.
* Attach any other additional information, like screenshots, you think may help.


### Do you want a new feature?

If there is a new feature you'd like to see in <img src="server/recwizard.png" alt="recwizard logo" width="15"/> Recwizard, please open an issue and describe:

1. What is the *motivation* behind this feature? Is it related to a problem or frustration with the library? Is it a feature related to something you need for a project? Is it something you worked on and think it could benefit the community?

   Whatever it is, we'd love to hear about it!

2. Describe your requested feature in as much detail as possible. The more you can tell us about it, the better we'll be able to help you.
3. Provide a *code snippet* that demonstrates the features usage.
4. If the feature is related to a paper, please include a link.

If your issue is well written we're already 80% of the way there by the time you create it.

## Do you want to implement a new model?

New models are constantly released and if you want to implement a new model, please provide the following information

* A short description of the model and a link to the paper.
* Link to the implementation if it is open-sourced.
* Link to the model weights if they are available.

If you are willing to contribute the model yourself, let us know so we can help you add it to Recwizard!

## Do you want to add documentation?

We're always looking for improvements to the documentation that make it more clear and accurate. Please let us know how the documentation can be improved such as typos and any content that is missing, unclear or inaccurate. We'll be happy to make the changes or help you make a contribution if you're interested!

For more details about how to generate, build, and write the documentation, take a look at the documentation [README](doc_link).

## Create a Pull Request

Before writing any code, we strongly advise you to search through the existing PRs or issues to make sure nobody is already working on the same thing. If you are unsure, it is always a good idea to open an issue to get some feedback.

You will need basic `git` proficiency to contribute to <img src="server/recwizard.png" alt="recwizard logo" width="15"/> Recwizard. While `git` is not the easiest tool to use, it has the greatest manual. Type `git --help` in a shell and enjoy! If you prefer books, [Pro Git](https://git-scm.com/book/en/v2) is a very good reference.

You'll need **[Python 3.10] or above to contribute to Recwizard. Follow the steps below to start contributing:

1. Fork the [repository](https://github.com/McAuley-Lab/RecWizard) by clicking on the **[Fork](https://github.com/McAuley-Lab/RecWizard/fork)** button on the repository's page. This creates a copy of the code under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

   ```bash
   git clone git@github.com:<your Github handle>/RecWizard.git
   cd RecWizard
   git remote add upstream https://github.com/huggingface/RecWizard.git
   ```

3. Create a new branch to hold your development changes:

   ```bash
   git checkout -b a-descriptive-name-for-my-changes
   ```

   🚨 **Do not** work on the `main` branch!

4. Set up a development environment by running the following command in a virtual environment:

   ```bash
   pip install -e ".[dev]"
   ```

   If Recwizard was already installed in the virtual environment, remove
   it with `pip uninstall Recwizard` before reinstalling it in editable
   mode with the `-e` flag.

5. Develop the features in your branch.

   As you work on your code, you should make sure the test suite
   passes. Run the tests impacted by your changes like this:

   ```bash
   pytest tests/<TEST_TO_RUN>.py
   ```

   Our code is based on huggingface transformers. For more information about tests, check out the  [Huggingface Testing](https://huggingface.co/docs/transformers/testing) guide.


Once you're happy with your changes, add the changed files with `git add` and record your changes locally with `git commit`:

   ```bash
   git add modified_file.py
   git commit
   ```

   Please remember to write [good commit
   messages](https://chris.beams.io/posts/git-commit/) to clearly communicate the changes you made!

   To keep your copy of the code up to date with the original
   repository, rebase your branch on `upstream/branch` *before* you open a pull request or if requested by a maintainer:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

   Push your changes to your branch:

   ```bash
   git push -u origin a-descriptive-name-for-my-changes
   ```

   If you've already opened a pull request, you'll need to force push with the `--force` flag. Otherwise, if the pull request hasn't been opened yet, you can just push your changes normally.

6. Now you can go to your fork of the repository on GitHub and click on **Pull Request** to open a pull request. Make sure you tick off all the boxes on our [checklist](https://github.com/McAuley-Lab/RecWizard/blob/main/CONTRIBUTING.md/#pull-request-checklist) below. When you're ready, you can send your changes to the project maintainers for review.

7. It's ok if maintainers request changes, it happens to our core contributors
   too! So everyone can see the changes in the pull request, work in your local
   branch and push the changes to your fork. They will automatically appear in
   the pull request.

### Pull request checklist

☐ The pull request title should summarize your contribution.<br>
☐ If your pull request addresses an issue, please mention the issue number in the pull request description to make sure they are linked (and people viewing the issue know you are working on it).<br>
☐ To indicate a work in progress please prefix the title with `[WIP]`. These are
useful to avoid duplicated work, and to differentiate it from PRs ready to be merged.<br>
☐ Make sure existing tests pass.<br>
☐ If adding a new feature, also add tests for it.<br>
   - If you are adding a new model, make sure you use `ModelTester.all_model_classes = (MyModel, MyModelWithLMHead,...)` to trigger the common tests.
   - If you are adding new `@slow` tests, make sure they pass using
     `RUN_SLOW=1 python -m pytest tests/models/my_new_model/test_my_new_model.py`.
   - If you are adding a new tokenizer, write tests and make sure
     `RUN_SLOW=1 python -m pytest tests/models/{your_model_name}/test_tokenization_{your_model_name}.py` passes.
   - CircleCI does not run the slow tests, but GitHub Actions does every night!<br>

☐ All public methods must have informative docstrings


### Style guide

For documentation strings, Recwizard follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
