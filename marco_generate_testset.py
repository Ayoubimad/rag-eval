from langchain_community.document_loaders import DirectoryLoader, TextLoader
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona


def main():
    num_samples = 1500

    path = "./data/arxiv/arxiv_papers_md_cleaned"
    loader = DirectoryLoader(
        path,
        glob=f"*.md",
        loader_cls=TextLoader,
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents from {path}")

    llm = ChatOpenAI(
        api_key="foo",
        base_url="http://172.18.21.137:8000/v1",
        model="ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g",
    )
    embeddings = OpenAIEmbeddings(
        api_key="foo", base_url="http://172.18.21.126:8000", model="BGE-M3"
    )

    generator_llm = LangchainLLMWrapper(llm)
    generator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    p1 = Persona(
        name="IT Researcher",
        role_description="""An IT researcher that wants to know information about scientific papers and research in the field of IT.""",
    )

    p2 = Persona(
        name="Computer Science Student",
        role_description="""
        A university student studying Computer Science who needs to understand technical concepts, algorithms, and research findings for their coursework and projects. They often ask questions about implementations, theoretical foundations, and practical applications.""",
    )

    p3 = Persona(
        name="Technology Teacher",
        role_description="""A high school or college teacher who teaches computer science and technology subjects. They need to explain complex concepts in simpler terms and are interested in finding educational examples and practical demonstrations for their students.""",
    )

    p4 = Persona(
        name="Tech Enthusiast",
        role_description="""A general technology enthusiast who is interested in understanding new developments in computer science and IT, but might not have formal technical education. They tend to ask questions focused on practical applications and high-level concepts.""",
    )

    persona_list = [p1, p2, p3, p4]

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        persona_list=persona_list,
    )

    dataset = generator.generate_with_langchain_docs(
        docs, testset_size=num_samples, raise_exceptions=False
    )
    with open("dataset_gen_arxiv.pkl", "wb") as f:
        import pickle

        pickle.dump(dataset, f)
    print("[+] Generazione completata.")
    dataset.to_csv(
        f"./data_sport/ragas_testset_arxiv_papers.csv",
    )


if __name__ == "__main__":
    main()
