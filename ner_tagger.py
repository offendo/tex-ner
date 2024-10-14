import os
import random
import string

import math
import numpy as np
import pandas as pd

# from conlleval import evaluate as conllevaluate
from argparse import ArgumentParser
from tqdm import tqdm
from itertools import takewhile
from pathlib import Path

from sklearn.metrics import precision_recall_fscore_support

parser = ArgumentParser("crf")

parser.add_argument("cmd", help="train|test", choices=["train", "test"])

# Training stuff
parser.add_argument("--name", "-n", help="name of run", required=True, type=str)
parser.add_argument("--epochs", "-e", help="num training epochs", default=10, type=int)
parser.add_argument("--data", "-d", help="path to data directory", default="crf_data/", type=str)
parser.add_argument("--optimizer", help="optimize using sgd", choices=["sgd", "adagrad"], default="sgd")
parser.add_argument("--lr", help="adagrad learning rate", type=float, default=1e-2)

# Testing stuff
parser.add_argument("--model", "-m", help="path to model", required=False, default="model")
parser.add_argument("--input", "-i", help="path to input file", required=False, default=None)

args = parser.parse_args()

directory = os.path.expanduser(f"~/src/autoformalization/ner/crf_data/{args.name}")
Path(directory).mkdir(exist_ok=True, parents=True)


def create_multiclass_labels(
    definition: bool = False,
    theorem: bool = False,
    proof: bool = False,
    example: bool = False,
    name: bool = False,
    reference: bool = False,
    start: bool = True,
    stop: bool = True,
):
    """Nilay's way to generate the label-to-ID map, depending on which tags we're working with."""
    class_names = tuple(
        k
        for k, v in dict(
            definition=definition,
            theorem=theorem,
            proof=proof,
            example=example,
            name=name,
            reference=reference,
        ).items()
        if v
    )

    labels = []
    for arg in sorted(class_names):
        new = []
        for label in labels:
            new.append(f"{label}-{arg}")
        new.append(arg)
        labels.extend(new)

    labels = ["O"] + labels
    for lab in labels:
        if "name" in lab and "reference" in lab:
            labels.remove(lab)

    if start:
        labels += ["<START>"]
    if stop:
        labels += ["<STOP>"]
    return {lab: idx for idx, lab in enumerate(labels)}


LABEL2ID = create_multiclass_labels(definition=True, theorem=True, proof=True, example=True)
TAGS = list(LABEL2ID.keys())


def decode(input_length, tagset, score):
    """
    Compute the highest scoring sequence according to the scoring function.
    :param input_length: int. number of tokens in the input including <START> and <STOP>
    :param tagset: Array of strings, which are the possible tags.  Does not have <START>, <STOP>
    :param score: function from current_tag (string), previous_tag (string), i (int) to the score.  i=0 points to
        <START> and i=1 points to the first token. i=input_length-1 points to <STOP>
    :return: Array strings of length input_length, which is the highest scoring tag sequence including <START> and <STOP>
    """
    # Look at the function compute_score for an example of how the tag sequence should be scored

    viterbi = [[] for i in range(input_length)]
    backpointers = [[] for i in range(input_length)]
    num_tag = len(tagset)

    def max_prev(t, s):
        # t is index in sentence, s is cur_tag (string)
        m = viterbi[t - 1][0] + score(s, tagset[0], t)
        mindex = 0
        for i in range(1, len(tagset)):
            v = viterbi[t - 1][i] + score(s, tagset[i], t)
            if v > m:
                mindex = i
                m = v
        return m, mindex

    # Base case
    for s in range(num_tag):
        temp = score(tagset[s], "<START>", 1)
        viterbi[1].append(temp)

    # print(viterbi[1])

    # Recursive
    for t in range(2, input_length - 1):
        for s in range(num_tag):
            cur_max, sPrev = max_prev(t, tagset[s])
            viterbi[t].append(cur_max)
            backpointers[t].append(sPrev)
        # print(viterbi[t])

    # Termination
    _, best_tag = max_prev(input_length - 1, "<STOP>")
    # print(best_tag)

    # Follow backpointers
    best_path = [0 for i in range(input_length)]
    best_path[input_length - 1] = "<STOP>"
    # print("input_length="+str(input_length))
    for t in reversed(range(2, input_length - 1)):
        # print("t="+str(t))
        best_path[t] = tagset[best_tag]
        best_tag = backpointers[t][best_tag]
    best_path[1] = tagset[best_tag]
    best_path[0] = "<START>"
    return best_path


def compute_score(tag_seq, input_length, score):
    """
    Computes the total score of a tag sequence
    :param tag_seq: Array of String of length input_length. The tag sequence including <START> and <STOP>
    :param input_length: Int. input length including the padding <START> and <STOP>
    :param score: function from current_tag (string), previous_tag (string), i (int) to the score.  i=0 points to
        <START> and i=1 points to the first token. i=input_length-1 points to <STOP>
    :return:
    """
    total_score = 0
    for i in range(1, input_length):
        total_score += score(tag_seq[i], tag_seq[i - 1], i)
    return total_score


def compute_features(tag_seq, input_length, features):
    """
    Compute f(xi, yi)
    :param tag_seq: [tags] already padded with <START> and <STOP>
    :param input_length: input length including the padding <START> and <STOP>
    :param features: func from token index to FeatureVector
    :return:
    """
    feats = FeatureVector({})
    for i in range(1, input_length):
        feats.times_plus_equal(1, features.compute_features(tag_seq[i], tag_seq[i - 1], i))
    return feats


def sgd(training_size, epochs, gradient, parameters, training_observer):
    """
    Stochastic gradient descent
    :param training_size: int. Number of examples in the training set
    :param epochs: int. Number of epochs to run SGD for
    :param gradient: func from index (int) in range(training_size) to a FeatureVector of the gradient
    :param parameters: FeatureVector.  Initial parameters.  Should be updated while training
    :param training_observer: func that takes epoch and parameters.  You can call this function at the end of each
           epoch to evaluate on a dev set and write out the model parameters for early stopping.
    :return: final parameters
    """
    # Look at the FeatureVector object.  You'll want to use the function times_plus_equal to update the
    # parameters.
    # To implement early stopping you can call the function training_observer at the end of each epoch.
    i = 0

    epoch_bar = tqdm(list(range(1, epochs + 1)), position=0, leave=True)
    for i in epoch_bar:
        epoch_bar.set_description(f"Epoch {i}")
        data_indices = [i for i in range(training_size)]
        random.shuffle(data_indices)
        pbar = tqdm(data_indices, position=1, leave=False)
        for t in pbar:
            parameters.times_plus_equal(-1, gradient(t))
        f1 = training_observer(i, parameters)
        print(f"\nTest F1: {f1}")
        epoch_bar.set_postfix({"f1": f1})
        epoch_bar.refresh()

    return parameters


def adagrad(training_size, epochs, learning_rate, gradient, parameters, training_observer):
    """
    Stochastic gradient descent
    :param training_size: int. Number of examples in the training set
    :param epochs: int. Number of epochs to run SGD for
    :param gradient: func from index (int) in range(training_size) to a FeatureVector of the gradient
    :param parameters: FeatureVector.  Initial parameters.  Should be updated while training
    :param training_observer: func that takes epoch and parameters.  You can call this function at the end of each
           epoch to evaluate on a dev set and write out the model parameters for early stopping.
    :return: final parameters
    """
    # Look at the FeatureVector object.  You'll want to use the function times_plus_equal to update the
    # parameters.
    # To implement early stopping you can call the function training_observer at the end of each epoch.
    i = 0
    sumSq = FeatureVector({})
    avg_weights = FeatureVector({})

    while i < epochs:
        print("i=" + str(i))
        data_indices = [i for i in range(training_size)]
        random.shuffle(data_indices)
        counter = 0
        for t in data_indices:
            grad = gradient(t)
            for key, value in grad.fdict.items():
                sumSq.fdict[key] = sumSq.fdict.get(key, 0) + value * value
                parameters.fdict[key] = parameters.fdict.get(key, 0) - learning_rate * value / math.sqrt(
                    sumSq.fdict[key]
                )
            counter += 1
        i += 1
        training_observer(i, parameters)
    return parameters


def train(data, tagset, epochs):
    """
    Trains the model on the data and returns the parameters
    :param data: Array of dictionaries representing the data.  One dictionary for each data point (as created by the
        make_data_point function).
    :param tagset: Array of Strings.  The list of tags.
    :param epochs: Int. The number of epochs to train
    :return: FeatureVector. The learned parameters.
    """
    parameters = FeatureVector({})

    def perceptron_gradient(i):
        """
        Computes the gradient of the Perceptron loss for example i
        :param i: Int
        :return: FeatureVector
        """
        inputs = data[i]
        input_len = len(inputs["tokens"])
        gold_labels = inputs["labels"]
        features = Features(inputs)

        def score(cur_tag, pre_tag, i):
            return parameters.dot_product(features.compute_features(cur_tag, pre_tag, i))

        tags = decode(input_len, tagset, score)
        fvector = compute_features(tags, input_len, features)  # Add the predicted features
        # print('Input:', inputs)        # helpful for debugging
        # print("Predicted Feature Vector:", fvector.fdict)
        # print("Predicted Score:", parameters.dot_product(fvector))
        fvector.times_plus_equal(
            -1, compute_features(gold_labels, input_len, features)
        )  # Subtract the features for the gold labels
        # print("Gold Labels Feature Vector: ", compute_features(gold_labels, input_len, features).fdict)
        # print("Gold Labels Score:", parameters.dot_product(compute_features(gold_labels, input_len, features)))
        return fvector

    def training_observer(epoch, parameters):
        """
        Evaluates the parameters on the development data, and writes out the parameters to a 'model.iter'+epoch and
        the predictions to 'ner.dev.out'+epoch.
        :param epoch: int.  The epoch
        :param parameters: Feature Vector.  The current parameters
        :return: Double. F1 on the development data
        """
        if not os.path.exists(f"{args.data}/ner.test"):
            return None
        dev_data = read_data(f"{args.data}/ner.test")
        _, _, f1 = evaluate(dev_data, parameters, tagset)
        write_predictions("ner.dev.out" + str(epoch), dev_data, parameters, tagset)
        parameters.write_to_file("model.iter" + str(epoch))
        return f1

    if args.optimizer == "sgd":
        return sgd(len(data), epochs, perceptron_gradient, parameters, training_observer)
    elif args.optimizer == "adagrad":
        return adagrad(
            len(data),
            epochs,
            learning_rate=args.learning_rate,
            gradient=perceptron_gradient,
            parameters=parameters,
            training_observer=training_observer,
        )


def predict(inputs, input_len, parameters, tagset):
    """

    :param inputs:
    :param input_len:
    :param parameters:
    :param tagset:
    :return:
    """
    features = Features(inputs)

    def score(cur_tag, pre_tag, i):
        return parameters.dot_product(features.compute_features(cur_tag, pre_tag, i))

    return decode(input_len, tagset, score)


def make_data_point(sent):
    """
        Creates a dictionary from String to an Array of Strings representing the data.  The dictionary items are:
        dic['tokens'] = Tokens padded with <START> and <STOP>
        dic['pos'] = POS tags padded with <START> and <STOP>
        dic['NP_chunk'] = Tags indicating noun phrase chunks, padded with <START> and <STOP>
        dic['labels'] = The gold tags padded with <START> and <STOP>
    :param sent: String.  The input CoNLL format string
    :return: Dict from String to Array of Strings.
    """
    dic = {}
    sent = [s.strip().split() for s in sent]
    dic["tokens"] = ["<START>"] + [s[0] for s in sent] + ["<STOP>"]
    dic["pos"] = ["<START>"] + [s[1] for s in sent] + ["<STOP>"]
    dic["NP_chunk"] = ["<START>"] + [s[2] for s in sent] + ["<STOP>"]
    dic["labels"] = ["<START>"] + [s[3] for s in sent] + ["<STOP>"]
    return dic


def read_data_old(filename):
    """
    Reads the CoNLL 2003 data into an array of dictionaries (a dictionary for each data point).
    :param filename: String
    :return: Array of dictionaries.  Each dictionary has the format returned by the make_data_point function.
    """
    data = []
    with open(directory + "/" + filename, "r") as f:
        sent = []
        for line in f.readlines():
            if line.strip():
                sent.append(line)
            else:
                data.append(make_data_point(sent))
                sent = []
        data.append(make_data_point(sent))

    return data


def read_data(filename: str):
    df = pd.read_json(filename)

    df["tokens"] = df.tokens.apply(lambda x: ["<START>", *x, "<STOP>"])
    df["preds"] = df.preds.apply(lambda x: ["<START>", *x, "<STOP>"])

    if "labels" in df.columns:
        df["labels"] = df.labels.apply(lambda x: ["<START>", *x, "<STOP>"])
    if "logits" in df.columns:
        mod_logs = lambda ls: [l + [0.0, 0.0] for l in ls]  # add the start/stop logits
        start_logits = [0] * (len(df.logits[0][0]))
        stop_logits = [0] * (len(df.logits[0][0]))
        df["logits"] = df.logits.apply(lambda x: [start_logits, *x, stop_logits]).apply(mod_logs)

    return df.to_dict(orient="records")


def write_predictions(out_filename, all_inputs, parameters, tagset):
    """
    Writes the predictions on all_inputs to out_filename, in CoNLL 2003 evaluation format.
    Each line is token, pos, NP_chuck_tag, gold_tag, predicted_tag (separated by spaces)
    Sentences are separated by a newline
    The file can be evaluated using the command: python conlleval.py < out_file
    :param out_filename: filename of the output
    :param all_inputs:
    :param parameters:
    :param tagset:
    :return:
    """
    # raise NotImplementedError("Rewrite this function")
    with open(directory + "/" + out_filename.replace("/", "."), "w", encoding="utf-8") as f:
        for inputs in all_inputs:
            input_len = len(inputs["tokens"])
            tag_seq = predict(inputs, input_len, parameters, tagset)
            for i, tag in enumerate(tag_seq[1:-1]):  # deletes <START> and <STOP>
                f.write(
                    " ".join(
                        [
                            inputs["tokens"][i + 1],
                            inputs["preds"][i + 1],
                            inputs["labels"][i + 1] if "labels" in inputs else "",
                            tag,
                            # inputs["logits"][i + 1],
                        ]
                    )
                    + "\n"
                )  # i + 1 because of <START>
            f.write("\n")


def evaluate(data, parameters, tagset):
    """
    Evaluates precision, recall, and F1 of the tagger compared to the gold standard in the data
    :param data: Array of dictionaries representing the data.  One dictionary for each data point (as created by the
        make_data_point function)
    :param parameters: FeatureVector.  The model parameters
    :param tagset: Array of Strings.  The list of tags.
    :return: Tuple of (prec, rec, f1)
    """
    all_labels = []
    all_predicted_tags = []
    for inputs in data:
        all_labels.extend(inputs["labels"][1:-1])  # deletes <START> and <STOP>
        input_len = len(inputs["tokens"])
        all_predicted_tags.extend(predict(inputs, input_len, parameters, tagset)[1:-1])  # deletes <START> and <STOP>
    p, r, f, _ = precision_recall_fscore_support(all_labels, all_predicted_tags, average="micro")
    return p, r, f


def test_decoder1():
    tagset = ["NN", "VB"]  # make up our own tagset

    def score_wrap(cur_tag, pre_tag, i):
        retval = score(cur_tag, pre_tag, i)
        print("Score(" + cur_tag + "," + pre_tag + "," + str(i) + ") returning " + str(retval))
        return retval

    def score(cur_tag, pre_tag, i):
        if i == 0:
            print("ERROR: Don't call score for i = 0 (that points to <START>, with nothing before it)")
        if i == 1:
            if pre_tag != "<START>":
                print("ERROR: Previous tag should be <START> for i = 1. Previous tag = " + pre_tag)
            if cur_tag == "NN":
                return 6
            if cur_tag == "VB":
                return 4
        if i == 2:
            if cur_tag == "NN" and pre_tag == "NN":
                return 4
            if cur_tag == "NN" and pre_tag == "VB":
                return 9
            if cur_tag == "VB" and pre_tag == "NN":
                return 5
            if cur_tag == "VB" and pre_tag == "VB":
                return 0
        if i == 3:
            if cur_tag != "<STOP>":
                print("ERROR: Current tag at i = 3 should be <STOP>. Current tag = " + cur_tag)
            if pre_tag == "NN":
                return 1
            if pre_tag == "VB":
                return 1

    predicted_tag_seq = decode(4, tagset, score_wrap)
    print("Predicted tag sequence should be = <START> VB NN <STOP>")
    print("Predicted tag sequence = " + " ".join(predicted_tag_seq))
    print(
        "Score of ['<START>','VB','NN','<STOP>'] = " + str(compute_score(["<START>", "VB", "NN", "<STOP>"], 4, score))
    )
    print("Max score should be = 14")
    print("Max score = " + str(compute_score(predicted_tag_seq, 4, score)))


def test_decoder():
    tagset = ["N", "V"]  # make up our own tagset

    def score(cur_tag, pre_tag, i):
        if i == 0:  # <START>
            print("ERROR: Don't call score for i = 0 (that points to <START>, with nothing before it)")
        if i == 1:  # they
            if pre_tag != "<START>":
                print("ERROR: Previous tag should be <START> for i = 1. Previous tag = " + pre_tag)
            if cur_tag == "N":
                return -2
            if cur_tag == "V":
                return -10
        if i == 2:  # can
            if cur_tag == "N" and pre_tag == "N":
                return 4
            if cur_tag == "N" and pre_tag == "V":
                return 9
            if cur_tag == "V" and pre_tag == "N":
                return 5
            if cur_tag == "V" and pre_tag == "V":
                return 0
        if i == 3:  # fish
            if cur_tag == "N" and pre_tag == "N":
                return 4
            if cur_tag == "N" and pre_tag == "V":
                return 9
            if cur_tag == "V" and pre_tag == "N":
                return 5
            if cur_tag == "V" and pre_tag == "V":
                return 0
        if i == 4:  # <STOP>
            if cur_tag != "<STOP>":
                print("ERROR: Current tag at i = 4 should be <STOP>. Current tag = " + cur_tag)
            if pre_tag == "N":
                return 1
            if pre_tag == "V":
                return 1
        print("ERROR: ")

    predicted_tag_seq = decode(5, tagset, score)
    print("Predicted tag sequence should be = <START> VB NN <STOP>")
    print("Predicted tag sequence = " + " ".join(predicted_tag_seq))
    print("Max score should be = 14")
    print("Max score = " + str(compute_score(predicted_tag_seq, 5, score)))


def main_predict(data_filename, model_filename):
    """
    Main function to make predictions.
    Loads the model file and runs the NER tagger on the data, writing the output in CoNLL 2003 evaluation format to data_filename.out
    :param data_filename: String
    :param model_filename: String
    :return: None
    """
    data = read_data(data_filename)
    parameters = FeatureVector({})
    parameters.read_from_file(model_filename)

    tagset = TAGS

    write_predictions(data_filename + ".out", data, parameters, tagset)
    evaluate(data, parameters, tagset)

    return


def main_train():
    """
    Main function to train the model
    :return: None
    """
    print("Reading training data")
    train_data = read_data(f"{args.data}/ner.train")
    # train_data = read_data('ner.train')[1:1] # if you want to train on just one example

    tagset = TAGS

    print("Training...")
    parameters = train(train_data, tagset, epochs=args.epochs)
    print("Training done")
    # dev_data = read_data("ner.dev")
    # evaluate(dev_data, parameters, tagset)
    test_data = read_data(f"{args.data}/ner.test")
    evaluate(test_data, parameters, tagset)
    parameters.write_to_file("model")

    return


class Features(object):
    def __init__(self, inputs):
        """
        Creates a Features object
        :param inputs: Dictionary from String to an Array of Strings.
            Created in the make_data_point function.
            inputs['tokens'] = Tokens padded with <START> and <STOP>
            inputs['pos'] = POS tags padded with <START> and <STOP>
            inputs['NP_chunk'] = Tags indicating noun phrase chunks, padded with <START> and <STOP>
            inputs['labels'] = DON'T USE! The gold tags padded with <START> and <STOP>
        """
        self.inputs = inputs
        self.label2id = LABEL2ID
        self.sequence_lengths, self.sequence_positions = self.compute_sequence_lengths(self.inputs["preds"])
        self.equation_bounds = self.compute_equation_bounds(self.inputs["tokens"])

    @staticmethod
    def compute_sequence_lengths(preds):
        lengths = []
        pos = []
        i = 0
        while i < len(preds):
            tok = preds[i]
            seqlen = len(list(takewhile(lambda x: x == tok, preds[i:])))
            lengths += [seqlen] * seqlen
            pos += list(range(1, seqlen + 1))
            i += seqlen
        return lengths, pos

    @staticmethod
    def compute_equation_bounds(tokens):
        i = 0
        equation_toks = []
        in_equation = False
        while i < len(tokens):
            tok = tokens[i]
            if tok == r"Ġ\(":
                in_equation = True
                equation_toks.append(True)
            elif tok == r"\)":
                equation_toks.append(True)
                in_equation = False
            else:
                equation_toks.append(in_equation)
            i += 1
        return equation_toks

    def compute_features(self, cur_tag, pre_tag, i):
        """
        Computes the local features for the current tag, the previous tag, and position i
        :param cur_tag: String.  The current tag.
        :param pre_tag: String.  The previous tag.
        :param i: Int. The position
        :return: FeatureVector

        * [x] BERT log prob of the predicted tag and is the tag the same
        * [x] Is BERT prob >= .99 for predicted tag
        * [x] def_tag + Total length of the current seq
        * [x] Predicted current tag
        * [x] Predicted prev tag
        * [x] Predicted next tag
        * [x] In equation
        * [x] Is punct
        * [ ] Is .
        * [ ] Starts with G
        * [ ] Starts with Let
        * [ ] Starts with Example
        * [ ] Distance from nearest "Example keyword" etc
        * [ ] Position in the run of tags
        """
        feats = FeatureVector({})

        # Current tag and previous tag
        feats.times_plus_equal(1, FeatureVector({f"t={cur_tag}": 1}))
        feats.times_plus_equal(1, FeatureVector({f"ti={cur_tag}+ti-1={pre_tag}": 1}))

        # Max prob for tag
        same = cur_tag == self.inputs["preds"][i]
        softmax = lambda x: np.exp(x) / np.exp(x).sum(axis=-1).reshape(-1)
        max_prob_gt_99 = max(softmax(np.array(self.inputs["logits"][i]))) > 0.99
        feats.times_plus_equal(1, FeatureVector({"same+BProbgt.99": int(same and max_prob_gt_99)}))
        feats.times_plus_equal(1, FeatureVector({f"t={cur_tag}+same+BProbgt.99": int(same and max_prob_gt_99)}))

        # Sequence of predictions
        pred_cur = self.inputs["preds"][i]
        feats.times_plus_equal(1, FeatureVector({f"t={cur_tag}+p={pred_cur}": 1}))
        if i - 1 >= 0:
            pred_prev = self.inputs["preds"][i - 1]
            feats.times_plus_equal(1, FeatureVector({f"t={cur_tag}+pi-1={pred_prev}": 1}))
        if i + 1 < len(self.inputs["preds"]):
            pred_next = self.inputs["preds"][i + 1]
            feats.times_plus_equal(1, FeatureVector({f"t={cur_tag}+pi+1={pred_next}": 1}))

        # Current tag's BERT log-probability
        prob_of_cur_tag = self.inputs["logits"][i][self.label2id[cur_tag]]
        feats.times_plus_equal(1, FeatureVector({"logp(t)=": prob_of_cur_tag}))

        # # Sequence length of current prediction (log length)
        # feats.times_plus_equal(1, FeatureVector({"same+len=": math.log(self.sequence_lengths[i]) if same else 0}))
        # feats.times_plus_equal(1, FeatureVector({"t={cur_tag}+len({pred_cur})=": math.log(self.sequence_lengths[i])}))

        # # Sequence length of current prediction (bucketed)
        # if self.sequence_lengths[i] < 2:
        #     feats.times_plus_equal(1, FeatureVector({f"len({pred_cur})<2": 0.1}))
        # elif self.sequence_lengths[i] < 5:
        #     feats.times_plus_equal(1, FeatureVector({f"len({pred_cur})<5": 0.1}))
        # elif self.sequence_lengths[i] < 10:
        #     feats.times_plus_equal(1, FeatureVector({f"len({pred_cur})<10": 0.1}))
        # else:
        #     feats.times_plus_equal(1, FeatureVector({f"len({pred_cur})>10": 0.1}))

        # # Position in current run
        # percent_complete = self.sequence_positions[i] / self.sequence_lengths[i]
        # feats.times_plus_equal(1, FeatureVector({f"pos=": percent_complete}))
        # feats.times_plus_equal(1, FeatureVector({f"same+pos=": percent_complete if same else 0}))

        # # Is in equation?
        # feats.times_plus_equal(1, FeatureVector({"in_eq": int(self.equation_bounds[i])}))

        # # Is it a punctuation?
        # ispunct = all([c in string.punctuation for c in self.inputs["tokens"][i]])
        # feats.times_plus_equal(1, FeatureVector({"ispunct": int(ispunct)}))

        return feats


class FeatureVector(object):

    def __init__(self, fdict):
        self.fdict = fdict

    def times_plus_equal(self, scalar, v2):
        """
        self += scalar * v2
        :param scalar: Double
        :param v2: FeatureVector
        :return: None
        """
        for key, value in v2.fdict.items():
            self.fdict[key] = scalar * value + self.fdict.get(key, 0)

    def dot_product(self, v2):
        """
        Computes the dot product between self and v2.  It is more efficient for v2 to be the smaller vector (fewer
        non-zero entries).
        :param v2: FeatureVector
        :return: Int
        """
        retval = 0
        for key, value in v2.fdict.items():
            retval += value * self.fdict.get(key, 0)
        return retval

    def write_to_file(self, filename):
        """
        Writes the feature vector to a file.
        :param filename: String
        :return: None
        """
        # print("Writing to " + filename)
        with open(directory + "/" + filename, "w", encoding="utf-8") as f:
            for key, value in self.fdict.items():
                f.write("{} {}\n".format(key, value))

    def read_from_file(self, filename):
        """
        Reads a feature vector from a file.
        :param filename: String
        :return: None
        """
        self.fdict = {}
        with open(filename, "r") as f:
            for line in f.readlines():
                txt = line.split()
                self.fdict[txt[0]] = float(txt[1])


# test_decoder1()
# test_decoder()
if args.cmd == "train":
    main_train()
elif args.cmd == "test":
    main_predict(args.input, args.model)
