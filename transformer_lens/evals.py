"""Evaluation Helpers.

This module contains some rough evals for models, but you are likely better off using the
HuggingFace evaluate library if you want to do anything properly. This is however here if you want
it and want to eg cheaply and roughly compare models you've trained to baselines.
"""

import random
from typing import Dict, List, Optional, Union

import einops
import torch
import tqdm.auto as tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from transformer_lens import utils


# %%
def sanity_check(model):
    """
    Very basic eval - just feeds a string into the model (in this case, the first paragraph of Circuits: Zoom In), and returns the loss. It's a rough and quick sanity check - if the loss is <5 the model is probably OK, if the loss is >7 something's gone wrong.

    Note that this is a very basic eval, and doesn't really tell you much about the model's performance.
    """

    text = "Many important transition points in the history of science have been moments when science 'zoomed in.' At these points, we develop a visualization or tool that allows us to see the world in a new level of detail, and a new field of science develops to study the world through this lens."

    return model(text, return_type="loss")


# %%
def make_wiki_data_loader(tokenizer, batch_size=8):
    """
    Evaluate on Wikitext 2, a dump of Wikipedia articles. (Using the train set because it's larger, I don't really expect anyone to bother with quarantining the validation set nowadays.)

    Note there's likely to be dataset leakage into training data (though I believe GPT-2 was explicitly trained on non-Wikipedia data)
    """
    wiki_data = load_dataset("wikitext", "wikitext-2-v1", split="train")
    print(len(wiki_data))
    dataset = utils.tokenize_and_concatenate(wiki_data, tokenizer)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return data_loader


def make_owt_data_loader(tokenizer, batch_size=8):
    """
    Evaluate on OpenWebText an open source replication of the GPT-2 training corpus (Reddit links with >3 karma)

    I think the Mistral models were trained on this dataset, so they get very good performance.
    """
    owt_data = load_dataset("stas/openwebtext-10k", split="train")
    print(len(owt_data))
    dataset = utils.tokenize_and_concatenate(owt_data, tokenizer)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return data_loader


def make_pile_data_loader(tokenizer, batch_size=8):
    """
    Evaluate on the first 10k texts from The Pile.

    The Pile is EleutherAI's general-purpose english dataset, made of 22 subsets
    including academic papers, books, internet content...
    """
    pile_data = load_dataset("NeelNanda/pile-10k", split="train")
    print(len(pile_data))
    dataset = utils.tokenize_and_concatenate(pile_data, tokenizer)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return data_loader


def make_code_data_loader(tokenizer, batch_size=8):
    """
    Evaluate on the CodeParrot dataset, a dump of Python code.

    All models seem to get significantly lower loss here (even non-code trained models like GPT-2),
    presumably code is much easier to predict than natural language?
    """
    code_data = load_dataset("codeparrot/codeparrot-valid-v2-near-dedup", split="train")
    print(len(code_data))
    dataset = utils.tokenize_and_concatenate(code_data, tokenizer, column_name="content")
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return data_loader


# All 57 subjects available in the MMLU benchmark
MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

MMLU_ANSWER_LETTERS = ["A", "B", "C", "D"]


def make_mmlu_data_loader(
    subjects: Optional[Union[str, List[str]]] = None,
    split: str = "test",
    num_samples: Optional[int] = None,
):
    """
    Load MMLU (Massive Multitask Language Understanding) dataset.

    MMLU tests model performance on 57 subjects across STEM, humanities, social sciences,
    and more. Each question is multiple choice with 4 options (A, B, C, D).

    Paper: https://arxiv.org/abs/2009.03300
    Dataset: https://huggingface.co/datasets/cais/mmlu

    Args:
        subjects: Subject(s) to evaluate on. Can be:
            - None: Use all 57 subjects (default)
            - str: Single subject name (e.g., "abstract_algebra")
            - List[str]: Multiple subjects
        split: Which split to use - "test", "validation", or "dev". Default is "test".
        num_samples: Optional limit on number of samples per subject. If None, uses all samples.

    Returns:
        List of dictionaries with MMLU examples, each containing:
            - "question": str
            - "choices": List[str] (4 choices)
            - "answer": int (0-3, correct choice index)
            - "subject": str

    Examples:

    .. code-block:: python

        >>> from transformer_lens.evals import make_mmlu_data_loader

        >>> # Load specific subject
        >>> mmlu_data = make_mmlu_data_loader(subjects="college_mathematics")

        >>> # Load multiple subjects
        >>> mmlu_data = make_mmlu_data_loader(
        ...     subjects=["abstract_algebra", "astronomy", "college_chemistry"]
        ... )
    """
    # Handle subjects parameter
    if subjects is None:
        subjects_to_load = MMLU_SUBJECTS
    elif isinstance(subjects, str):
        subjects_to_load = [subjects]
    else:
        subjects_to_load = list(subjects)

    # Validate subjects
    invalid_subjects = set(subjects_to_load) - set(MMLU_SUBJECTS)
    if invalid_subjects:
        raise ValueError(
            f"Invalid subject(s): {invalid_subjects}. "
            f"Valid subjects: {', '.join(sorted(MMLU_SUBJECTS))}"
        )

    # Load data for each subject
    mmlu_data = []
    for subject in subjects_to_load:
        try:
            # Load dataset for this subject
            dataset = load_dataset("cais/mmlu", subject, split=split)

            # Limit samples if requested
            samples_to_take = (
                len(dataset) if num_samples is None else min(num_samples, len(dataset))
            )

            # Convert to our format
            for i in range(samples_to_take):
                example = dataset[i]
                mmlu_data.append(
                    {
                        "question": example["question"],
                        "choices": example["choices"],
                        "answer": example["answer"],
                        "subject": subject,
                    }
                )
        except Exception as e:
            print(f"Warning: Could not load subject '{subject}': {e}")
            continue

    print(f"Loaded {len(mmlu_data)} MMLU examples from {len(subjects_to_load)} subject(s)")
    return mmlu_data


DATASET_NAMES = ["wiki", "owt", "pile", "code"]
DATASET_LOADERS = [
    make_wiki_data_loader,
    make_owt_data_loader,
    make_pile_data_loader,
    make_code_data_loader,
]


# %%
@torch.inference_mode()
def evaluate_on_dataset(model, data_loader, truncate=100, device="cuda"):
    running_loss = 0
    total = 0
    for batch in tqdm.tqdm(data_loader):
        loss = model(batch["tokens"].to(device), return_type="loss").mean()
        running_loss += loss.item()
        total += 1
        if total > truncate:
            break
    return running_loss / total


# %%
@torch.inference_mode()
def induction_loss(
    model, tokenizer=None, batch_size=4, subseq_len=384, prepend_bos=None, device="cuda"
):
    """
    Generates a batch of random sequences repeated twice, and measures model performance on the second half. Tests whether a model has induction heads.

    By default, prepends a beginning of string token (when prepend_bos flag defaults to None, model.cfg.default_prepend_bos is used
    whose default is True unless specified otherwise), which is useful to give models a resting position, and sometimes models were trained with this.
    """
    # Make the repeated sequence
    first_half_tokens = torch.randint(100, 20000, (batch_size, subseq_len)).to(device)
    repeated_tokens = einops.repeat(first_half_tokens, "b p -> b (2 p)")

    # Use the provided prepend_bos as an override if it's not None;
    # otherwise use model.cfg.default_prepend_bos (defaults to True)
    prepend_bos = utils.override_or_use_default_value(
        model.cfg.default_prepend_bos, override=prepend_bos
    )

    # Prepend a Beginning Of String token
    if prepend_bos:
        if tokenizer is None:
            tokenizer = model.tokenizer
        repeated_tokens[:, 0] = tokenizer.bos_token_id
    # Run the model, and extract the per token correct log prob
    logits = model(repeated_tokens, return_type="logits")
    correct_log_probs = utils.lm_cross_entropy_loss(logits, repeated_tokens, per_token=True)
    # Take the loss over the second half of the sequence
    return correct_log_probs[:, subseq_len + 1 :].mean()


# %%
@torch.inference_mode()
def evaluate(model, truncate=100, batch_size=8, tokenizer=None):
    if tokenizer is None:
        tokenizer = model.tokenizer
    losses = {}
    for data_name, data_loader_fn in zip(DATASET_NAMES, DATASET_LOADERS):
        data_loader = data_loader_fn(tokenizer=tokenizer, batch_size=batch_size)
        loss = evaluate_on_dataset(model, data_loader, truncate=truncate)
        print(f"{data_name}: {loss}")
        losses[f"{data_name}_loss"] = loss
    return losses


# %%
class IOIDataset(Dataset):
    """
    Dataset for Indirect Object Identification tasks.
    Paper: https://arxiv.org/pdf/2211.00593.pdf

    Example:

    .. code-block:: python

        >>> from transformer_lens.evals import ioi_eval, IOIDataset
        >>> from transformer_lens.HookedTransformer import HookedTransformer

        >>> model = HookedTransformer.from_pretrained('gpt2-small')
        Loaded pretrained model gpt2-small into HookedTransformer

        >>> # Evaluate like this, printing the logit difference
        >>> print(round(ioi_eval(model, num_samples=100)["Logit Difference"], 3))
        5.476

        >>> # Can use custom dataset
        >>> ds = IOIDataset(
        ...     tokenizer=model.tokenizer,
        ...     num_samples=100,
        ...     templates=['[A] met with [B]. [B] gave the [OBJECT] to [A]'],
        ...     names=['Alice', 'Bob', 'Charlie'],
        ...     nouns={'OBJECT': ['ball', 'book']},
        ... )
        >>> print(round(ioi_eval(model, dataset=ds)["Logit Difference"], 3))
        5.397
    """

    def __init__(
        self,
        tokenizer,
        templates: Optional[List[str]] = None,
        names: Optional[List[str]] = None,
        nouns: Optional[Dict[str, List[str]]] = None,
        num_samples: int = 1000,
        symmetric: bool = False,
        prepend_bos: bool = True,
    ):
        self.tokenizer = tokenizer
        self.prepend_bos = prepend_bos

        self.templates = templates if templates is not None else self.get_default_templates()
        self.names = names if names is not None else self.get_default_names()
        self.nouns = nouns if nouns is not None else self.get_default_nouns()

        self.samples = []
        for _ in range(num_samples // 2 if symmetric else num_samples):
            # If symmetric, get_sample will return two samples
            self.samples.extend(self.get_sample(symmetric=symmetric))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = self.tokenizer.encode(sample["text"])
        if self.prepend_bos:
            prompt = [self.tokenizer.bos_token_id] + prompt

        return {
            "prompt": torch.LongTensor(prompt),
            "IO": torch.LongTensor(self.tokenizer.encode(sample["IO"])),
            "S": torch.LongTensor(self.tokenizer.encode(sample["S"])),
        }

    def get_sample(self, symmetric=False) -> List[Dict[str, str]]:
        random.seed(42)
        template: str = random.choice(self.templates)
        for noun_type, noun_list in self.nouns.items():
            template = template.replace(f"[{noun_type}]", random.choice(noun_list))

        samples: List[Dict[str, str]] = []

        # Sample two names without replacement
        names = random.sample(self.names, 2)
        sample = template.replace("[A]", names[0])
        sample = sample.replace("[B]", names[1])
        # Prepend spaces to IO and S so that the target is e.g. " Mary" and not "Mary"
        samples.append({"text": sample, "IO": " " + names[0], "S": " " + names[1]})

        if symmetric:
            sample_2 = template.replace("[A]", names[1])
            sample_2 = sample_2.replace("[B]", names[0])
            samples.append({"text": sample_2, "IO": " " + names[1], "S": " " + names[0]})

        return samples

    @staticmethod
    def get_default_names():
        return ["John", "Mary"]

    @staticmethod
    def get_default_templates():
        return [
            "[A] and [B] went to the [LOCATION] to buy [OBJECT]. [B] handed the [OBJECT] to [A]",
            "Then, [B] and [A] went to the [LOCATION]. [B] gave the [OBJECT] to [A]",
        ]

    @staticmethod
    def get_default_nouns():
        return {
            "LOCATION": ["store", "market"],
            "OBJECT": ["milk", "eggs", "bread"],
        }


@torch.inference_mode()
def ioi_eval(model, dataset=None, batch_size=8, num_samples=1000, tokenizer=None, symmetric=False):
    """Evaluate the Model on the Indirect Object Identification Task.

    Args:
        model: HookedTransformer model.
        dataset: PyTorch Dataset that returns a dict with keys "prompt", "IO", and "S".
        batch_size: Batch size to use.
        num_samples: Number of samples to use.
        tokenizer: Tokenizer to use.
        symmetric: Whether to use the symmetric version of the task.

    Returns:
        Average logit difference and accuracy.
    """
    if tokenizer is None:
        tokenizer = model.tokenizer

    if dataset is None:
        dataset = IOIDataset(tokenizer, num_samples=num_samples, symmetric=symmetric)

    def collate(samples):
        prompts = [sample["prompt"] for sample in samples]
        padded_prompts = torch.nn.utils.rnn.pad_sequence(prompts, batch_first=True)
        return {
            "prompt": padded_prompts,
            "IO": [sample["IO"] for sample in samples],
            "S": [sample["S"] for sample in samples],
            "prompt_length": [p.shape[0] for p in prompts],
        }

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    total_correct = 0
    total_logit_diff = 0
    for batch in tqdm.tqdm(data_loader):
        batch_logits = model(batch["prompt"], return_type="logits")

        for i in range(batch_logits.shape[0]):
            io = batch["IO"][i]
            s = batch["S"][i]
            prefix_length = batch["prompt_length"][i] - io.shape[0]

            # Trim io and s to the same length
            min_len = min(io.shape[0], s.shape[0])
            io = io[:min_len]
            s = s[:min_len]

            # Remove identical prefixes
            start_idx = torch.where(io != s)[0][0]
            io = io[start_idx]
            s = s[start_idx]
            logit_idx = prefix_length + start_idx - 1

            # Get the logits for the tokens we care about
            logits = batch_logits[i, logit_idx]
            correct_logit = logits[io]
            incorrect_logit = logits[s]

            # Compute stats
            logit_diff = correct_logit - incorrect_logit
            correct = logit_diff > 0
            total_correct += correct.item()
            total_logit_diff += logit_diff.item()

    return {
        "Logit Difference": total_logit_diff / len(dataset),
        "Accuracy": total_correct / len(dataset),
    }


@torch.inference_mode()
def mmlu_eval(
    model,
    tokenizer=None,
    subjects: Optional[Union[str, List[str]]] = None,
    split: str = "test",
    num_samples: Optional[int] = None,
    device: str = "cuda",
):
    """Evaluate a model on the MMLU benchmark.

    MMLU (Massive Multitask Language Understanding) is a benchmark for evaluating language models
    on 57 subjects across STEM, humanities, social sciences, and more. Each question is
    multiple-choice with 4 options.

    For each question, all four answer choices (A-D) are shown in the prompt and the model's
    log probability for each answer letter token is compared. This is a zero-shot evaluation;
    standard MMLU benchmarks typically use 5-shot prompting for higher accuracy.

    Paper: https://arxiv.org/abs/2009.03300

    Args:
        model: HookedTransformer model to evaluate.
        tokenizer: Tokenizer to use. If None, uses model.tokenizer.
        subjects: Subject(s) to evaluate on. Can be None (all 57 subjects), a single subject
            string, or a list of subjects. See :const:`MMLU_SUBJECTS` for valid names.
        split: Which split to use - "test", "validation", or "dev". Default is "test".
        num_samples: Optional limit on number of samples per subject. If None, uses all samples.
        device: Device to run evaluation on. Default is "cuda".

    Returns:
        Dictionary containing:
            - "accuracy": Overall accuracy (0-1)
            - "num_correct": Number of correct predictions
            - "num_total": Total number of questions
            - "subject_scores": Dict mapping subject names to their accuracy

    Examples:

    .. code-block:: python

        >>> from transformer_lens import HookedTransformer
        >>> from transformer_lens.evals import mmlu_eval

        >>> model = HookedTransformer.from_pretrained("gpt2-small")
        Loaded pretrained model gpt2-small into HookedTransformer

        >>> # Evaluate on a specific subject
        >>> results = mmlu_eval(model, subjects="abstract_algebra", num_samples=10)
        >>> print(f"Accuracy: {results['accuracy']:.2%}")
        Accuracy: 30.00%

        >>> # Evaluate on multiple subjects
        >>> results = mmlu_eval(
        ...     model,
        ...     subjects=["astronomy", "college_mathematics"],
        ...     num_samples=20
        ... )
        >>> for subject, acc in results["subject_scores"].items():
        ...     print(f"{subject}: {acc:.2%}")
        astronomy: 35.00%
        college_mathematics: 25.00%
    """
    if tokenizer is None:
        tokenizer = model.tokenizer

    # Load MMLU data
    mmlu_data = make_mmlu_data_loader(subjects=subjects, split=split, num_samples=num_samples)

    if len(mmlu_data) == 0:
        raise ValueError("No MMLU data loaded. Check your subjects parameter.")

    # Precompute token IDs for answer letters A, B, C, D
    # Done once here instead of per-question for efficiency
    answer_letter_token_ids = []
    for letter in MMLU_ANSWER_LETTERS:
        # Try with space prefix first (how it appears after "Answer:")
        token_ids = tokenizer.encode(" " + letter, add_special_tokens=False)
        if len(token_ids) == 1:
            answer_letter_token_ids.append(token_ids[0])
        else:
            # Fallback to without space
            token_ids = tokenizer.encode(letter, add_special_tokens=False)
            answer_letter_token_ids.append(token_ids[0])

    # Track results
    num_correct = 0
    num_total = 0
    subject_correct: Dict[str, int] = {}
    subject_total: Dict[str, int] = {}

    # Process examples
    for example in tqdm.tqdm(mmlu_data, desc="Evaluating MMLU"):
        question = example["question"]
        choices = example["choices"]
        correct_answer = example["answer"]
        subject = example["subject"]

        # Initialize subject tracking
        if subject not in subject_correct:
            subject_correct[subject] = 0
            subject_total[subject] = 0

        # Format prompt with all choices shown (standard MMLU format)
        prompt = f"Question: {question}\n"
        prompt += "Choices:\n"
        for idx, choice_text in enumerate(choices):
            letter = chr(65 + idx)  # A, B, C, D
            prompt += f"{letter}. {choice_text}\n"
        prompt += "Answer:"

        # Tokenize the prompt
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Get logits
        logits = model(tokens, return_type="logits")

        # Get log probabilities at the last position (predicting the answer letter)
        last_log_probs = torch.nn.functional.log_softmax(logits[0, -1, :], dim=-1)

        # Score each answer choice by its letter token probability
        choice_log_probs = []
        for idx in range(len(choices)):
            token_id = answer_letter_token_ids[idx]
            choice_log_probs.append(last_log_probs[token_id].item())

        # Select the choice with highest log probability
        predicted_answer = choice_log_probs.index(max(choice_log_probs))

        # Check if correct
        is_correct = predicted_answer == correct_answer
        num_correct += int(is_correct)
        num_total += 1
        subject_correct[subject] += int(is_correct)
        subject_total[subject] += 1

    # Compute accuracies
    overall_accuracy = num_correct / num_total if num_total > 0 else 0.0
    subject_scores = {
        subject: subject_correct[subject] / subject_total[subject]
        for subject in subject_correct.keys()
    }

    return {
        "accuracy": overall_accuracy,
        "num_correct": num_correct,
        "num_total": num_total,
        "subject_scores": subject_scores,
    }
