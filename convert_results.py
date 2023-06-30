import csv
import sys
import os

import pandas as pd


def main(PATH: str):
    df = pd.read_csv(os.path.join(PATH, "test_pred.csv"))
    indices = pd.read_csv(os.path.join(PATH, "test_indices.txt"), header=None, names=["index"])

    df["problem_id"] = indices["index"].apply(lambda text: int(text))
    data = df.to_dict('records')

    converted_data = []
    current_index = int(data[0]["problem_id"])
    current_data_point = data[0]
    for row in data[1:]:
        index = int(row["problem_id"])
        if index != current_index:
            converted_data.append(current_data_point)
            current_data_point = row
            current_index = index
        else:
            current_data_point["Question"] += "\n" + row["Question"]
            current_data_point["Ground Truth"] += row["Ground Truth"]

            new_questions = row["Prediction"].split("\n")
            existing_questions = current_data_point["Prediction"].split("\n")
            for new_question in new_questions:
                if new_question not in existing_questions:
                    existing_questions.append(new_question)

            existing_questions = [question for question in existing_questions if len(question) > 0]
            current_data_point["Prediction"] = "\n".join(existing_questions)
    converted_data.append(current_data_point)

    PATH_CONVERTED_RESULTS = f"{os.path.join(PATH)}/converted_test_pred.csv"
    with open(PATH_CONVERTED_RESULTS, "w") as pred_file:
        write = csv.writer(pred_file)
        write.writerow(["Question", "Ground Truth", "Prediction"])
        for row in converted_data:
            write.writerow([row["Question"], row["Ground Truth"], row["Prediction"]])
    print(f"File saved at :{PATH_CONVERTED_RESULTS}")

    return PATH_CONVERTED_RESULTS


if __name__ == '__main__':
    # path to predictions and newline delimited file where each row refers to the index of algebra story problem
    PATH = sys.argv[1]
    main(PATH)
