from site import PREFIXES
import warnings
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer
import random
import re
import matplotlib.pyplot as plt
import random as rd
import copy

NAMES = [
    "Michael",
    "Christopher",
    "Jessica",
    "Matthew",
    "Ashley",
    "Jennifer",
    "Joshua",
    "Amanda",
    "Daniel",
    "David",
    "James",
    "Robert",
    "John",
    "Joseph",
    "Andrew",
    "Ryan",
    "Brandon",
    "Jason",
    "Justin",
    "Sarah",
    "William",
    "Jonathan",
    "Stephanie",
    "Brian",
    "Nicole",
    "Nicholas",
    "Anthony",
    "Heather",
    "Eric",
    "Elizabeth",
    "Adam",
    "Megan",
    "Melissa",
    "Kevin",
    "Steven",
    "Thomas",
    "Timothy",
    "Christina",
    "Kyle",
    "Rachel",
    "Laura",
    "Lauren",
    "Amber",
    "Brittany",
    "Danielle",
    "Richard",
    "Kimberly",
    "Jeffrey",
    "Amy",
    "Crystal",
    "Michelle",
    "Tiffany",
    "Jeremy",
    "Benjamin",
    "Mark",
    "Emily",
    "Aaron",
    "Charles",
    "Rebecca",
    "Jacob",
    "Stephen",
    "Patrick",
    "Sean",
    "Erin",
    "Jamie",
    "Kelly",
    "Samantha",
    "Nathan",
    "Sara",
    "Dustin",
    "Paul",
    "Angela",
    "Tyler",
    "Scott",
    "Katherine",
    "Andrea",
    "Gregory",
    "Erica",
    "Mary",
    "Travis",
    "Lisa",
    "Kenneth",
    "Bryan",
    "Lindsey",
    "Kristen",
    "Jose",
    "Alexander",
    "Jesse",
    "Katie",
    "Lindsay",
    "Shannon",
    "Vanessa",
    "Courtney",
    "Christine",
    "Alicia",
    "Cody",
    "Allison",
    "Bradley",
    "Samuel",
]

ABC_TEMPLATES = [
    "Then, [A], [B] and [C] went to the [PLACE]. [B] and [C] gave a [OBJECT] to [A]",
    "Afterwards [A], [B] and [C] went to the [PLACE]. [B] and [C] gave a [OBJECT] to [A]",
    "When [A], [B] and [C] arrived at the [PLACE], [B] and [C] gave a [OBJECT] to [A]",
    "Friends [A], [B] and [C] went to the [PLACE]. [B] and [C] gave a [OBJECT] to [A]",
]

BAC_TEMPLATES = [template.replace("[B]", "[A]", 1).replace("[A]", "[B]", 1) for template in ABC_TEMPLATES]

BABA_TEMPLATES = [
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument, and afterwards [B] said to [A]",
    "After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
    "While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
    "While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
    "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument. Afterwards [B] said to [A]",
    "The [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to [A]",
    "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
]

ABBA_TEMPLATES = BABA_TEMPLATES[:]

for i in range(len(ABBA_TEMPLATES)):
    first_clause = True
    for j in range(1, len(ABBA_TEMPLATES[i]) - 1):
        if ABBA_TEMPLATES[i][j - 1 : j + 2] == "[B]" and first_clause:
            ABBA_TEMPLATES[i] = ABBA_TEMPLATES[i][:j] + "A" + ABBA_TEMPLATES[i][j + 1 :]
        elif ABBA_TEMPLATES[i][j - 1 : j + 2] == "[A]" and first_clause:
            first_clause = False
            ABBA_TEMPLATES[i] = ABBA_TEMPLATES[i][:j] + "B" + ABBA_TEMPLATES[i][j + 1 :]

VERBS = [" tried", " said", " decided", " wanted", " gave"]
PLACES = [
    "store",
    "garden",
    "restaurant",
    "school",
    "hospital",
    "office",
    "house",
    "station",
]
OBJECTS = [
    "ring",
    "kiss",
    "bone",
    "basketball",
    "computer",
    "necklace",
    "drink",
    "snack",
]

ANIMALS = [
    "dog",
    "cat",
    "snake",
    "elephant",
    "beetle",
    "hippo",
    "giraffe",
    "tiger",
    "husky",
    "lion",
    "panther",
    "whale",
    "dolphin",
    "beaver",
    "rabbit",
    "fox",
    "lamb",
    "ferret",
]


def multiple_replace(dict, text):
    # from: https://stackoverflow.com/questions/15175142/how-can-i-do-multiple-substitutions-using-regex
    # Create a regular expression from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict[mo.string[mo.start() : mo.end()]], text)


def iter_sample_fast(iterable, samplesize):
    results = []
    # Fill in the first samplesize elements:
    try:
        for _ in range(samplesize):
            results.append(next(iterable))
    except StopIteration:
        raise ValueError("Sample larger than population.")
    random.shuffle(results)  # Randomize their positions

    return results


NOUNS_DICT = NOUNS_DICT = {"[PLACE]": PLACES, "[OBJECT]": OBJECTS}


def gen_prompt_uniform(templates, names, nouns_dict, N, symmetric, prefixes=None, abc=False):
    nb_gen = 0
    ioi_prompts = []
    while nb_gen < N:
        temp = rd.choice(templates)
        temp_id = templates.index(temp)
        name_1 = ""
        name_2 = ""
        name_3 = ""
        while len(set([name_1, name_2, name_3])) < 3:
            name_1 = rd.choice(names)
            name_2 = rd.choice(names)
            name_3 = rd.choice(names)

        nouns = {}
        for k in nouns_dict:
            nouns[k] = rd.choice(nouns_dict[k])
        prompt = temp
        for k in nouns_dict:
            prompt = prompt.replace(k, nouns[k])

        if prefixes is not None:
            L = rd.randint(30, 40)
            pref = ".".join(rd.choice(prefixes).split(".")[:L])
            pref += "<|endoftext|>"
        else:
            pref = ""

        prompt1 = prompt.replace("[A]", name_1)
        prompt1 = prompt1.replace("[B]", name_2)
        if abc:
            prompt1 = prompt1.replace("[C]", name_3)
        prompt1 = pref + prompt1
        ioi_prompts.append({"text": prompt1, "IO": name_1, "S": name_2, "TEMPLATE_IDX": temp_id})
        if abc:
            ioi_prompts[-1]["C"] = name_3

        nb_gen += 1

        if symmetric and nb_gen < N:
            prompt2 = prompt.replace("[A]", name_2)
            prompt2 = prompt2.replace("[B]", name_1)
            prompt2 = pref + prompt2
            ioi_prompts.append({"text": prompt2, "IO": name_2, "S": name_1, "TEMPLATE_IDX": temp_id})
            nb_gen += 1
    return ioi_prompts


def gen_flipped_prompts(prompts, names, flip=("S2", "IO")):
    """_summary_

    Args:
        prompts (List[D]): _description_
        flip (tuple, optional): First element is the string to be replaced, Second is what to replace with. Defaults to ("S2", "IO").

    Returns:
        _type_: _description_
    """
    flipped_prompts = []

    for prompt in prompts:
        t = prompt["text"].split(" ")
        prompt = prompt.copy()
        if flip[0] == "S2":
            if flip[1] == "IO":
                t[len(t) - t[::-1].index(prompt["S"]) - 1] = prompt["IO"]
                temp = prompt["IO"]
                prompt["IO"] = prompt["S"]
                prompt["S"] = temp
            elif flip[1] == "RAND":
                rand_name = names[np.random.randint(len(names))]
                while rand_name == prompt["IO"] or rand_name == prompt["S"]:
                    rand_name = names[np.random.randint(len(names))]
                t[len(t) - t[::-1].index(prompt["S"]) - 1] = rand_name
            else:
                raise ValueError("Invalid flip[1] value")

        elif flip[0] == "IO":
            if flip[1] == "RAND":
                rand_name = names[np.random.randint(len(names))]
                while rand_name == prompt["IO"] or rand_name == prompt["S"]:
                    rand_name = names[np.random.randint(len(names))]

                t[t.index(prompt["IO"])] = rand_name
                t[t.index(prompt["IO"])] = rand_name
                prompt["IO"] = rand_name
            elif flip[1] == "ANIMAL":
                rand_animal = ANIMALS[np.random.randint(len(ANIMALS))]
                t[t.index(prompt["IO"])] = rand_animal
                prompt["IO"] = rand_animal
                # print(t)
            elif flip[1] == "S1":
                io_index = t.index(prompt["IO"])
                s1_index = t.index(prompt["S"])
                io = t[io_index]
                s1 = t[s1_index]
                t[io_index] = s1
                t[s1_index] = io
            else:
                raise ValueError("Invalid flip[1] value")

        elif flip[0] in ["S", "S1"]:
            if flip[1] == "ANIMAL":
                new_s = ANIMALS[np.random.randint(len(ANIMALS))]
            if flip[1] == "RAND":
                new_s = names[np.random.randint(len(names))]
            t[t.index(prompt["S"])] = new_s
            if flip[0] == "S":  # literally just change the first S if this is S1
                t[len(t) - t[::-1].index(prompt["S"]) - 1] = new_s
                prompt["S"] = new_s
        elif flip[0] == "END":
            if flip[1] == "S":
                t[len(t) - t[::-1].index(prompt["IO"]) - 1] = prompt["S"]
        elif flip[0] == "PUNC":
            n = []

            # separate the punctuation from the words
            for i, word in enumerate(t):
                if "." in word:
                    n.append(word[:-1])
                    n.append(".")
                elif "," in word:
                    n.append(word[:-1])
                    n.append(",")
                else:
                    n.append(word)

            # remove punctuation, important that you check for period first
            if flip[1] == "NONE":
                if "." in n:
                    n[n.index(".")] = ""
                elif "," in n:
                    n[len(n) - n[::-1].index(",") - 1] = ""

            # remove empty strings
            while "" in n:
                n.remove("")

            # add punctuation back to the word before it
            while "," in n:
                n[n.index(",") - 1] += ","
                n.remove(",")

            while "." in n:
                n[n.index(".") - 1] += "."
                n.remove(".")

            t = n

        elif flip[0] == "C2":
            if flip[1] == "A":
                t[len(t) - t[::-1].index(prompt["C"]) - 1] = prompt["A"]
        elif flip[0] == "S+1":
            if t[t.index(prompt["S"]) + 1] == "and":
                t[t.index(prompt["S"]) + 1] = ["with one friend named", "accompanied by"][np.random.randint(2)]
            else:
                t[t.index(prompt["S"]) + 1] = (
                    t[t.index(prompt["S"])] + ", after a great day, " + t[t.index(prompt["S"]) + 1]
                )
                del t[t.index(prompt["S"])]
        else:
            raise ValueError(f"Invalid flipper {flip[0]}")

        if "IO" in prompt:
            prompt["text"] = " ".join(t)
            flipped_prompts.append(prompt)
        else:
            flipped_prompts.append(
                {
                    "A": prompt["A"],
                    "B": prompt["B"],
                    "C": prompt["C"],
                    "text": " ".join(t),
                }
            )

    return flipped_prompts


# *Tok Idxs Methods


def get_name_idxs(prompts, tokenizer, idx_types=["IO", "S", "S2"]):
    name_idx_dict = dict((idx_type, []) for idx_type in idx_types)
    double_s2 = False
    for prompt in prompts:
        t = prompt["text"].split(" ")
        toks = tokenizer.tokenize(" ".join(t[:-1]))
        for idx_type in idx_types:
            if "2" in idx_type:
                idx = len(toks) - toks[::-1].index(tokenizer.tokenize(" " + prompt[idx_type[:-1]])[0]) - 1
            else:
                idx = toks.index(tokenizer.tokenize(" " + prompt[idx_type])[0])
            name_idx_dict[idx_type].append(idx)
        if "S" in idx_types and "S2" in idx_types:
            if name_idx_dict["S"][-1] == name_idx_dict["S2"][-1]:
                double_s2 = True
    if double_s2:
        warnings.warn("S2 index has been computed as the same for S and S2")

    return [torch.tensor(name_idx_dict[idx_type]) for idx_type in idx_types]


def get_end_idxs(prompts, tokenizer, name_tok_len=1):
    toks = torch.Tensor(tokenizer([prompt["text"] for prompt in prompts], padding=True).input_ids).type(torch.int)
    end_idxs = torch.tensor(
        [(toks[i] == 50256).nonzero()[0][0].item() if 50256 in toks[i] else toks.shape[1] for i in range(toks.shape[0])]
    )
    end_idxs = end_idxs - 1 - name_tok_len  # YOURE LOOKING AT TO NOT FINAL IO TOKEN
    return end_idxs


def get_rand_idxs(end_idxs, exclude):
    rand_idxs = []
    for i in range(len(end_idxs)):
        idx = np.random.randint(end_idxs[i])

        while idx in torch.vstack(exclude)[:, i]:
            idx = np.random.randint(end_idxs[i])
        rand_idxs.append(idx)
    return rand_idxs


def get_word_idxs(prompts, word_list, tokenizer):
    """Get the index of the words in word_list in the prompts. Exactly one of the word_list word has to be present in each prompt"""
    idxs = []
    tokenized_words = [tokenizer.decode(tokenizer(word)["input_ids"][0]) for word in word_list]
    for pr_idx, prompt in enumerate(prompts):
        toks = [
            tokenizer.decode(t) for t in tokenizer(prompt["text"], return_tensors="pt", padding=True)["input_ids"][0]
        ]
        idx = None
        for i, w_tok in enumerate(tokenized_words):
            if word_list[i] in prompt["text"]:
                try:
                    idx = toks.index(w_tok)
                    if toks.count(w_tok) > 1:
                        idx = len(toks) - toks[::-1].index(w_tok) - 1
                except:
                    idx = toks.index(w_tok)
                    # raise ValueError(toks, w_tok, prompt["text"])
        if idx is None:
            raise ValueError(f"Word {word_list} and {i} not found {prompt}")
        idxs.append(idx)
    return torch.tensor(idxs)


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ALL_SEM = [
    "S",
    "IO",
    "S2",
    "end",
    "S+1",
    "and",
]  # , "verb", "starts", "S-1", "punct"] # Kevin's antic averages


def get_idx_dict(ioi_prompts, tokenizer):
    (
        IO_idxs,
        S_idxs,
        S2_idxs,
    ) = get_name_idxs(ioi_prompts, tokenizer, idx_types=["IO", "S", "S2"])

    end_idxs = get_end_idxs(ioi_prompts, tokenizer, name_tok_len=1)
    rand_idxs = get_rand_idxs(end_idxs, exclude=[IO_idxs, S_idxs, S2_idxs])
    punc_idxs = get_word_idxs(
        ioi_prompts, [",", "."], tokenizer
    )  # if there is "," and '.' in the prompt, only the '.' index will be kept.
    verb_idxs = get_word_idxs(ioi_prompts, VERBS, tokenizer)
    # and_idxs = get_word_idxs(ioi_prompts, [" and"], tokenizer)

    return {
        "IO": IO_idxs,
        "IO-1": IO_idxs - 1,
        "IO+1": IO_idxs + 1,
        "S": S_idxs,
        "S-1": S_idxs - 1,
        "S+1": S_idxs + 1,
        "S2": S2_idxs,
        "end": end_idxs,  # the " to" token, the last one.
        "rand": rand_idxs,  # random index at each
        "punct": punc_idxs,
        "verb": verb_idxs,
        # "and": and_idxs,
        "starts": torch.zeros_like(verb_idxs),
    }


PREFIXES = [
    "             Afterwards,",
    "            Two friends met at a bar. Then,",
    "  After a long day,",
    "  After a long day,",
    "    Then,",
    "         Then,",
]


def flip_prefixes(ioi_prompts):
    ioi_prompts = copy.deepcopy(ioi_prompts)
    for prompt in ioi_prompts:
        if prompt["text"].startswith("The "):
            prompt["text"] = "After the lunch, the" + prompt["text"][4:]
        else:
            io_idx = prompt["text"].index(prompt["IO"])
            s_idx = prompt["text"].index(prompt["S"])
            first_idx = min(io_idx, s_idx)
            prompt["text"] = rd.choice(PREFIXES) + " " + prompt["text"][first_idx:]

    return ioi_prompts


def flip_names(ioi_prompts):
    ioi_prompts = copy.deepcopy(ioi_prompts)
    for prompt in ioi_prompts:
        punct_idx = max(
            [i for i, x in enumerate(list(prompt["text"])) if x in [",", "."]]
        )  # only flip name in the first clause
        io = prompt["IO"]
        s = prompt["S"]
        prompt["text"] = (
            prompt["text"][:punct_idx].replace(io, "#").replace(s, "@").replace("#", s).replace("@", io)
        ) + prompt["text"][punct_idx:]
        # print(prompt["text"])

    return ioi_prompts


class IOIDataset:
    def __init__(
        self,
        prompt_type: str,
        N=500,
        tokenizer=None,
        prompts=None,
        symmetric=False,
        prefixes=None,
        nb_templates=None,
        ioi_prompts_for_word_idxs=None,
    ):
        """
        ioi_prompts_for_word_idxs:
            if you want to use a different set of prompts to get the word indices, you can pass it here
            (example use case: making a ABCA dataset)
        """

        assert not (symmetric and prompt_type == "ABC")
        assert (prompts is not None) or (not symmetric) or (N % 2 == 0), f"{symmetric} {N}"
        assert nb_templates is None or (nb_templates % 2 == 0 or prompt_type != "mixed")
        self.prompt_type = prompt_type

        if nb_templates is None:
            nb_templates = len(BABA_TEMPLATES)

        if prompt_type == "ABBA":
            self.templates = ABBA_TEMPLATES[:nb_templates].copy()
        elif prompt_type == "BABA":
            self.templates = BABA_TEMPLATES[:nb_templates].copy()
        elif prompt_type == "mixed":
            self.templates = BABA_TEMPLATES[: nb_templates // 2].copy() + ABBA_TEMPLATES[: nb_templates // 2].copy()
            random.shuffle(self.templates)
        elif prompt_type == "ABC":
            self.templates = ABC_TEMPLATES[:nb_templates].copy()
        elif prompt_type == "BAC":
            self.templates = BAC_TEMPLATES[:nb_templates].copy()
        elif prompt_type == "ABC mixed":
            self.templates = ABC_TEMPLATES[: nb_templates // 2].copy() + BAC_TEMPLATES[: nb_templates // 2].copy()
            random.shuffle(self.templates)
        else:
            raise ValueError(prompt_type)

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer

        self.prefixes = prefixes
        self.prompt_type = prompt_type
        if prompts is None:
            self.ioi_prompts = gen_prompt_uniform(  # a list of dict of the form {"text": "Alice and Bob bla bla. Bob gave bla to Alice", "IO": "Alice", "S": "Bob"}
                self.templates,
                NAMES,
                nouns_dict={"[PLACE]": PLACES, "[OBJECT]": OBJECTS},
                N=N,
                symmetric=symmetric,
                prefixes=self.prefixes,
                abc=(prompt_type in ["ABC", "ABC mixed", "BAC"]),
            )
        else:
            assert N == len(prompts), f"{N} and {len(prompts)}"
            self.ioi_prompts = prompts

        all_ids = [prompt["TEMPLATE_IDX"] for prompt in self.ioi_prompts]
        all_ids_ar = np.array(all_ids)
        self.groups = []
        for id in list(set(all_ids)):
            self.groups.append(np.where(all_ids_ar == id)[0])

        small_groups = []
        for group in self.groups:
            if len(group) < 5:
                small_groups.append(len(group))
        if len(small_groups) > 0:
            warnings.warn(f"Some groups have less than 5 prompts, they have lengths {small_groups}")

        self.text_prompts = [prompt["text"] for prompt in self.ioi_prompts]  # a list of strings

        self.templates_by_prompt = []  # for each prompt if it's ABBA or BABA
        for i in range(N):
            if self.text_prompts[i].index(self.ioi_prompts[i]["IO"]) < self.text_prompts[i].index(
                self.ioi_prompts[i]["S"]
            ):
                self.templates_by_prompt.append("ABBA")
            else:
                self.templates_by_prompt.append("BABA")

        # print(self.ioi_prompts, "that's that")
        self.toks = torch.Tensor(
            self.tokenizer([prompt["text"] for prompt in self.ioi_prompts], padding=True).input_ids
        ).type(torch.int)

        if ioi_prompts_for_word_idxs is None:
            ioi_prompts_for_word_idxs = self.ioi_prompts
        self.word_idx = get_idx_dict(ioi_prompts_for_word_idxs, self.tokenizer)

        self.sem_tok_idx = {
            k: v for k, v in self.word_idx.items() if k in ALL_SEM
        }  # the semantic indices that kevin uses
        self.N = N
        self.max_len = max([len(self.tokenizer(prompt["text"]).input_ids) for prompt in self.ioi_prompts])

        self.io_tokenIDs = [self.tokenizer.encode(" " + prompt["IO"])[0] for prompt in self.ioi_prompts]
        self.s_tokenIDs = [self.tokenizer.encode(" " + prompt["S"])[0] for prompt in self.ioi_prompts]

    def gen_flipped_prompts(self, flip):
        """Return a IOIDataset where the name to flip has been replaced by a random name."""
        assert isinstance(flip, tuple) or flip in [
            "prefix",
        ], f"{flip=} is not a tuple. Probably change to ('IO', 'RAND') or equivalent?"

        if flip == "prefix":
            flipped_prompts = flip_prefixes(self.ioi_prompts)
        else:
            if flip == ("IO", "S1"):
                flipped_prompts = gen_flipped_prompts(
                    self.ioi_prompts,
                    None,
                    flip,
                )
            elif flip == ("S2", "IO"):
                flipped_prompts = gen_flipped_prompts(
                    self.ioi_prompts,
                    None,
                    flip,
                )

            else:
                assert flip[1] == "RAND" and flip[0] in ["S", "RAND", "S2", "IO", "S1", "S+1"], flip
                flipped_prompts = gen_flipped_prompts(self.ioi_prompts, NAMES, flip)

        flipped_ioi_dataset = IOIDataset(
            prompt_type=self.prompt_type,
            N=self.N,
            tokenizer=self.tokenizer,
            prompts=flipped_prompts,
            prefixes=self.prefixes,
            ioi_prompts_for_word_idxs=flipped_prompts if flip[0] == "RAND" else None,
        )
        return flipped_ioi_dataset

    def copy(self):
        copy_ioi_dataset = IOIDataset(
            prompt_type=self.prompt_type,
            N=self.N,
            tokenizer=self.tokenizer,
            prompts=self.ioi_prompts.copy(),
            prefixes=self.prefixes.copy() if self.prefixes is not None else self.prefixes,
            ioi_prompts_for_word_idxs=self.ioi_prompts.copy(),
        )
        return copy_ioi_dataset

    def __getitem__(self, key):
        sliced_prompts = self.ioi_prompts[key]
        sliced_dataset = IOIDataset(
            prompt_type=self.prompt_type,
            N=len(sliced_prompts),
            tokenizer=self.tokenizer,
            prompts=sliced_prompts,
            prefixes=self.prefixes,
        )
        return sliced_dataset

    def __setitem__(self, key, value):
        raise NotImplementedError()

    def __delitem__(self, key):
        raise NotImplementedError()

    def __len__(self):
        return self.N
