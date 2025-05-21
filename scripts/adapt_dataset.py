import pandas as pd
import json

df = pd.read_csv("/home/e4user/rag-eval/datasets/ragas_testset_arxi_papers.csv")

data = {
    "user_input": df["user_input"].tolist(),
    "reference": df["reference"].tolist(),
    "reference_contexts": [[context] for context in df["reference_contexts"].tolist()],
}

with open("/home/e4user/rag-eval/datasets/ragas_testset_arxi_papers.json", "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
