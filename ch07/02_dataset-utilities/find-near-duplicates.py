
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import argparse
import json
from sklearn import __version__ as sklearn_version
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Sample JSON dataset
example_data = [
    {"instruction": "What is the capital of Italy?",
     "input": "", "output": "The capital of Italy is Rome."
     },
    {"instruction": "What's the capital city of Italy?",
     "input": "", "output": "The capital city is Rome."
     },
    {"instruction": "Identify the main verb in the sentence: 'The cat sleeps on the couch.'",
     "input": "", "output": "The verb is 'sleeps'."
     },
    {"instruction": "Identify the verb in the following sentence: The cat sleeps on the couch.",
     "input": "", "output": "The verb in the sentence is \"sleeps.\""
     },
    # ...
]


def find_near_duplicates(json_data, threshold=0.8, key="instruction"):
    """The higher the threshold, the more similar the texts have to be to match"""

    # Extract instructions
    text = [item[key] for item in json_data if item[key]]
    near_duplicates = []

    if not text:
        return near_duplicates

    # Vectorize the text data
    vectorizer = TfidfVectorizer(stop_words=None)
    tfidf_matrix = vectorizer.fit_transform(text)

    # Compute cosine similarity between each pair of entries
    cos_sim_matrix = cosine_similarity(tfidf_matrix)

    # Find pairs of near-duplicate instructions based on the threshold

    for i in range(len(cos_sim_matrix)):
        for j in range(i+1, len(cos_sim_matrix)):
            if cos_sim_matrix[i, j] > threshold:
                near_duplicates.append((json_data[i], json_data[j], cos_sim_matrix[i, j]))

    return near_duplicates


def find_and_print_new_duplicates(json_data):
    """
    Searches each key in the first JSON object for duplicates across a list of JSON objects.
    Prints the duplicates if found.
    """
    for key in json_data[0].keys():
        near_duplicates = find_near_duplicates(json_data, key=key)
        separator = 50 * '='
        print(f"\n\n{separator}\nSearching '{key}' for duplicates ...\n{separator}")
        if not near_duplicates:
            print("No duplicates found")
        else:
            for dup in near_duplicates:
                print(
                    f"Duplicate pair found with similarity {dup[2]:.2f}:\n"
                    f"1. {dup[0][key]}\n2. {dup[1][key]}\n"
                )


if __name__ == "__main__":
    print("scikit-learn version:", sklearn_version)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_file",
        type=str,
        help=("Path to the dataset JSON file")
    )
    args = parser.parse_args()
    if not args.json_file:
        json_data = example_data

    else:
        with open(args.json_file, "r") as file:
            json_data = json.load(file)

    find_and_print_new_duplicates(json_data)
