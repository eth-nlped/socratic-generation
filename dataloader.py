import copy
import json
import os
import random
import torch
import re
from text2digits import text2digits

random.seed(hash("setting random seeds") % 2 ** 32 - 1)
EQUATION = "equation"
OPERATOR = "operator"

operator_map = {
    "+": "+",
    "-": "-",
    "/": "/",
    "*": "*"
}


class GSMDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, input_seq, output_seq):
        self.max_source_length = 512
        self.max_target_length = 128
        self.tokenizer = tokenizer
        self.input_seq = input_seq
        self.output_seq = output_seq
        self.encoding = self.tokenizer(self.input_seq,
                                       padding='longest',
                                       max_length=self.max_source_length,
                                       truncation=True,
                                       return_tensors="pt")
        self.input_ids, self.attention_mask = self.encoding.input_ids, self.encoding.attention_mask

        self.target_encoding = self.tokenizer(self.output_seq,
                                              padding='longest',
                                              max_length=self.max_target_length,
                                              truncation=True)
        self.labels = self.target_encoding.input_ids

    def __len__(self):
        return len(self.input_seq)

    def __getitem__(self, idx):
        input_tokens = self.input_ids[idx]
        output_tokens = self.labels[idx]
        mask = self.attention_mask[idx]

        output_tokens = torch.tensor(output_tokens)
        output_tokens[output_tokens == self.tokenizer.pad_token_id] = -100

        return dict(inpt=input_tokens, att_mask=mask, lbl=output_tokens, raw_input=self.input_seq[idx],
                    raw_output=self.output_seq[idx])


def read_jsonl(split: str):
    """ Reads the input file and parses into a list of question answer pairs.

    Args:
        path (str): Train, test or dev split

    Returns:
        [list]: List of all question answer pairs in dict format.
    """
    path = f"data/{split}_socratic.jsonl"
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def text_to_digits(input_seq: str):
    """Converts text to its digit representation

    Args:
        input_seq [list]: [A list of input sequences]

    Returns:
        input_seq [list]: [A list of input sequences with text as digits]
    """

    t2d = text2digits.Text2Digits()

    try:
        input_seq = t2d.convert(input_seq)
    except:
        pass

    return input_seq


def sentence_planning_window(all_data: list):
    """Create input and output pairs for training the model
    Args:
        all_data ([list]): [all data points: list of dict]
    Returns:
        input_seq, output_seq [list]: [input and output sequence for training the model]
    """

    pairs = []
    for data in all_data:
        out = data["answer"].split("\n")
        ques, ans = [], []
        for samp in out:
            if len(samp.split(" ** ")) == 2:
                ques.append(samp.split(" ** ")[0])
                ans.append(samp.split(" ** ")[1])
        nums = [re.findall("(\d+=)|(\d+\s=)", " ".join(re.findall("<<\S+", sent))) for sent in ans]
        nums = [num[0][0].replace("=", "") for num in nums if num != []]

        for r in range(len(nums)):
            # data["question"] = text_to_digits(data["question"])
            if r == len(nums) - 1:
                pairs.append([data["question"], ques[r]])
            else:
                val = data["question"].find(nums[r])
                punkt = data["question"][val:].find(".")
                if punkt == -1:
                    pairs.append([data["question"][val + 1:], " ".join([ques[left] for left in range(r, len(nums))])])
                    break
                pairs.append([data["question"][:val + punkt + 1], ques[r]])
                data["question"] = data["question"][val + punkt + 1:]

    input_seq = [p[0] for p in pairs]
    output_seq = [p[1] for p in pairs]

    return input_seq, output_seq


def iterative_sentence_planning_with_separators(all_data: list, planning: str = "", reward: str = ""):
    """Create input and output pairs for training the model
    Args:
        all_data ([list]): [all data points: list of dict]
    Returns:
        input_seq, output_seq [list]: [input and output sequence for training the model]
    """

    pairs = []
    data_point_index = 0
    data_points_indices = []
    for data in all_data:
        data_point_index = data_point_index + 1
        out = data["answer"].split("\n")
        ques, ans = [], []
        for samp in out:
            if len(samp.split(" ** ")) == 2:
                ques.append(samp.split(" ** ")[0])
                ans.append(samp.split(" ** ")[1])
        equation_sequence = [" ".join(re.findall("<<\S+>>", sent)) for sent in ans]
        operator_sequence = [re.sub('[^-+*/<>]+', ' ', equation, 0, re.I).strip() for equation in equation_sequence]
        nums = [re.findall("(\d+=)|(\d+\s=)|(\d+\.\d+=)|(\d+\.\d+\s=)", " ".join(re.findall("<<\S+", sent))) for sent in
                ans]
        # filter matched groups
        matched_numers = []
        for found_pattern in nums:
            if len(found_pattern) > 0:
                for matched_number in found_pattern[0]:
                    if len(matched_number) > 0:
                        matched_numers.append(matched_number.replace("=", ""))
        nums = matched_numers

        new_operator_sequence = []
        for operator_string in operator_sequence:
            new_string = ""
            for char_in_operator in operator_string:
                new_string += operator_map.get(char_in_operator, char_in_operator)
            new_operator_sequence.append(new_string)
        operator_sequence = new_operator_sequence

        previous_split = 0
        equation_index = 0

        if reward != "qa" or reward == "combined":
            data["question"] = text_to_digits(data["question"])

        modified_question = data["question"]
        additional_information = ""
        for r in range(len(nums)):
            if planning == EQUATION:
                additional_information = " ".join(equation_sequence[equation_index:])
            elif planning == OPERATOR:
                additional_information = " ".join(operator_sequence[equation_index:])

            if r == len(nums) - 1:
                pairs.append([data["question"][
                              :previous_split] + " [SEP] " + modified_question + " [/SEP] " + additional_information,
                              ques[r]])
                data_points_indices.append(data_point_index)
            else:
                val = modified_question.find(nums[r])
                punkt = modified_question[val:].find(". ")
                split_point = val + punkt + 1
                if punkt == -1:
                    pairs.append([data["question"][
                                  :previous_split] + " [SEP] " + modified_question + " [/SEP] " + additional_information,
                                  " ".join([ques[left] for left in range(r, len(nums))])])
                    data_points_indices.append(data_point_index)
                    break

                if planning == EQUATION:
                    valid_equations = []
                    for eq in equation_sequence[equation_index:]:
                        is_match = re.search(f"[\+\-\*\/\<]{{1}}{nums[r]}[\+\-\*\/\<\=]{{1}}", eq)
                        if is_match is not None:
                            valid_equations.append(eq)
                            break
                    equation_index = equation_index + len(valid_equations)
                    additional_information = " ".join(valid_equations)
                elif planning == OPERATOR:
                    valid_equations = []
                    for eq in equation_sequence[equation_index:]:
                        is_match = re.search(f"[\+\-\*\/\<]{{1}}{nums[r]}[\+\-\*\/\<\=]{{1}}", eq)
                        if is_match is not None:
                            valid_equations.append(eq)
                            break
                    additional_information = " ".join(
                        operator_sequence[equation_index:equation_index + len(valid_equations)])
                    equation_index = equation_index + len(valid_equations)

                pairs.append([data["question"][:previous_split] + " [SEP] " + modified_question[
                                                                              :split_point] + " [/SEP] " + additional_information + modified_question[
                                                                                                                                    split_point:],
                              ques[r]])
                data_points_indices.append(data_point_index)
                modified_question = modified_question[split_point:]
                previous_split += split_point

    input_seq = [p[0] for p in pairs]
    output_seq = [p[1] for p in pairs]
    input_seq = [re.sub('\s+', ' ', sentence).strip() for sentence in input_seq]
    return input_seq, output_seq, data_points_indices


def randomise_operators(input_sentences: list, sequence_of_operators: list, random_number_of_operators: bool = False):
    """Randomise the operators based on the number of operators present in each sentence
    """
    random_op_seq = []
    if random_number_of_operators:
        print("Random number of operators")
        for _ in sequence_of_operators:
            number_of_operators = random.randrange(2, 8, 1)
            temp_dict = []
            for _ in range(number_of_operators):
                temp_dict.append(random.choice(['-', '+', '*', '/']))
            random_op_seq.append(" ".join(temp_dict))
    else:
        for seq in sequence_of_operators:
            number_of_operators = seq.count("+") + seq.count("-") + seq.count("*") + seq.count("/")
            temp_dict = []
            for iter in range(number_of_operators):
                temp_dict.append(random.choice(['-', '+', '*', '/']))
            random_op_seq.append(" ".join(temp_dict))

    input_seq = [seq_ques + " [SEP] " + seq_op for seq_ques, seq_op in zip(input_sentences, random_op_seq)]
    # sanity check
    assert (len(input_sentences) == len(random_op_seq))

    return input_seq


def planning_strategy(all_data: list, planning: str, reward: str = ""):
    """Extract all operators/equations in a sentence and return it.

    Args:
        input_seq ([list]): [all input sequences in the list format]

    Returns:
        new_sequence [list]: [extracted operators in the same list format]
    """

    input_sequences = [data["question"] for data in all_data]
    input_sentences = copy.deepcopy(input_sequences)

    if reward != "qa" or reward == "combined":
        for iteration in range(len(input_sentences)):
            input_sentences[iteration] = text_to_digits(input_sentences[iteration])  # converts text to digits

    output_sequences, output_answers = [], []
    for data in all_data:
        all_sent = data["answer"].split("\n")
        single_joined_ques, single_joined_ans = [], []
        for single_sent in all_sent:
            ques = single_sent.split("**")
            if len(ques) != 1:
                single_joined_ques.append(ques[0] + "\n")
                single_joined_ans.append(ques[1] + " ")
        output_sequences.append(" ".join(single_joined_ques))
        output_answers.append(" ".join(single_joined_ans))

    if planning == OPERATOR:
        new_sequence = [" ".join(re.findall("<<\S+>>", sent)) for sent in output_answers]
        new_sequence = [re.sub('[^-+*/<>]+', ' ', equation, 0, re.I).strip() for equation in new_sequence]

        random_operators = os.environ.get("RANDOM_OPERATORS",
                                          False)  # False by default. Only for experimentation and ablation studies
        if random_operators:
            print("Running with random operators")
            input_seq = randomise_operators(input_sentences, new_sequence,
                                            os.environ.get("RANDOM_NUMBER_OF_OPERATORS", False))
        else:
            input_seq = [seq_ques + " [SEP] " + seq_op for seq_ques, seq_op in zip(input_sentences, new_sequence)]
            # sanity check
            assert (len(input_sentences) == len(new_sequence))

    elif planning == EQUATION:
        new_sequence = [" ".join(re.findall("<<\S+>>", sent)) for sent in output_answers]
        input_seq = [seq_ques + " [SEP] " + seq_op for seq_ques, seq_op in zip(input_sentences, new_sequence)]
        # sanity check
        assert (len(input_sequences) == len(new_sequence))

    elif planning == "None":
        input_seq = input_sentences

    return input_seq, output_sequences


def get_input_output_ques_gen_seq(iterative_split: bool, split: str, planning: str, reward: str):
    """ Get input and output sequences in specified format for the task of question generation

    Args:
        path (str): Train test or dev split

    Returns:
        [list]: [list of input and output sequences]
    """

    all_data = read_jsonl(split)

    if iterative_split:
        input_seq, output_sequences, data_points_indices = iterative_sentence_planning_with_separators(all_data,
                                                                                                       planning, reward)
    else:
        input_seq, output_sequences = planning_strategy(all_data, planning, reward)
        # non-iterative version - each line is a separate algebra story problem
        data_points_indices = list(range(1, len(input_seq) + 1))

    # sanity check
    assert (len(input_seq) == len(output_sequences) == len(data_points_indices))

    return input_seq, output_sequences, data_points_indices


def clean_answer_calculation(out_seq):
    """Clean the calculation needed to do in the answer

    Args:
        out_seq ([list]): [answers for each question with context]

    Returns:
        out_seq ([list]): [cleaned answers for each question with context]
    """

    new_out_seq = [sent.replace(re.findall("<<\S+", sent)[0], "") if len(re.findall("<<\S+", sent)) != 0 else sent for
                   sent in out_seq]
    return new_out_seq


def get_input_output_qna_seq(split: str):
    """ Get input and output sequences in specified format for the task of question answering

    Args:
        path (str): Train test or dev split

    Returns:
        [list]: [list of input and output sequences]
    """

    all_data = read_jsonl(split)
    context_sequences = [" context: " + data["question"] for data in all_data]

    input_sequences = []
    output_sequences = []
    for data, context in zip(all_data, context_sequences):
        all_sent = data["answer"].split("\n")
        for single_sent in all_sent:
            ques = single_sent.split("**")
            if len(ques) != 1:
                input_sequences.append("question: " + ques[0] + context)
                output_sequences.append(ques[1])

    cleaned_out_seq = clean_answer_calculation(output_sequences)
    return input_sequences, cleaned_out_seq


def critic_data_preprocessing(data: list, planning: str) -> (list, list):
    input_sequences = [data["question"] for data in data]
    input_sentences = copy.deepcopy(input_sequences)

    for iteration in range(len(input_sentences)):
        input_sentences[iteration] = text_to_digits(input_sentences[iteration])  # converts text to digits

    output_answers = []
    for row in data:
        all_sent = row["answer"].split("\n")
        single_joined_ans = []
        for single_sent in all_sent:
            ques = single_sent.split("**")
            if len(ques) != 1:
                single_joined_ans.append(ques[1] + " ")
        output_answers.append(" ".join(single_joined_ans))

    if planning == OPERATOR:
        new_sequence = [" ".join(re.findall("<<\S+>>", sent)) for sent in output_answers]
        assert (len(input_sequences) == len(new_sequence))
        output_sequences = [re.sub('[^-+*/<>]+', ' ', equation, 0, re.I).strip() for equation in new_sequence]
    elif planning == EQUATION:
        new_sequence = [" ".join(re.findall("<<\S+>>", sent)) for sent in output_answers]
        assert (len(input_sequences) == len(new_sequence))
        output_sequences = [equation for equation in new_sequence]
        # sanity check
    else:
        raise NotImplementedError

    assert (len(input_sentences) == len(output_sequences))
    return input_sentences, output_sequences


def critic_data_prep(split: str, planning: str):
    all_data = read_jsonl(split)
    input_sequences, output_sequences = critic_data_preprocessing(all_data, planning)
    return input_sequences, output_sequences
