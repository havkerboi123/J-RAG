from datasets import load_dataset
from dotenv import load_dotenv
from ragas import EvaluationDataset

load_dotenv()


def load_eval_dataset() -> EvaluationDataset:
    dataset = load_dataset("vibrantlabsai/amnesty_qa", "english_v3")
    return EvaluationDataset.from_hf_dataset(dataset["eval"])


if __name__ == "__main__":
    eval_dataset = load_eval_dataset()
    print(eval_dataset)
    print(eval_dataset[0])
