# Jorge4PubMed
_llm-based retrieval of PubMed ID (PMID) from the text of biomedical articles_

![](imgs/jorge.png)

## Motivation
We use a RAG to ask questions about a scientific topic based on a corpus of ingested pdf, and we are given a response. When we read the response, we want to have the same experience that we have when we read a paper, that is, for each statement, we want to have its reference.

This can be complex, since all the RAG knows is the source filename and the (possibly) the text in the pdf which generated that citation.

Instead, you would like to **generate article-style citations, even better with a clickable PubMed link**, which is directly related to the PubMed ID (PMID) of the paper.

While this is extremely easy for humans (just open the paper and google its title), it can be a nightmare for llms, which are known to hallucinate a lot when asked to carry out this task.

**To make a concrete example, you want to go from this...**


> The insula sends neural efferents to cortical areas from which it receives reciprocal afferents projections [`1831.pdf`, `aug86.pdf`].
>
> - `1838.pdf`
> - `aug86.pdf`

**...to this**
> The insula sends neural efferents to cortical areas from which it receives reciprocal afferents projections [[Mesulam1982](https://pubmed.ncbi.nlm.nih.gov/7174907/), [Augustine1986](https://pubmed.ncbi.nlm.nih.gov/8957561/)].
>
> - Mesulam MM & Mufson (1982). Insula of the old world monkey. III: Efferent cortical output and comments on function. J Comp Neurol. [PubMed](https://pubmed.ncbi.nlm.nih.gov/7174907/)
> - Augustine JR. (1986). Circuitry and functional aspects of the insular lobe in primates including humans. Brain Res Brain Res Rev. [PubMed](https://pubmed.ncbi.nlm.nih.gov/8957561/)

<br>

Jorge - a deliberate reference to the blind librarian with encyclopedic knowledge from _The Name Of The Rose_ - can help you with that.


## How Jorge operates
- Extract the first lines of each paper and ask an llm to extract the title
- Make an initial search of the words in the titles a [broad proximity](https://www.ncbi.nlm.nih.gov/books/NBK25499/)
- Retrieve the PMID for succesful results
- Calculate the embeddings (using a [local model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) of the llm-retrieved and pubmed-retrieved titles, and reject the PMID for those titles where the cosine similarity is < 0.94
- Make a deep search using a sliding window of words for the titles which still do not have a PMID, this time using a more [restrictive proximity](https://stackoverflow.com/questions/77412057/searching-entrez-based-on-paper-title). Retain the PMID for which the max cosine similarity was > 0.94
- Use the PMIDs to retrieve the full citation and store everything into a SQLite database


## Running Jorge on your papers
Currently Jorge expects to find papers in markdown format. This can be easily (and quickly) achieved using [marker-pdf](https://github.com/VikParuchuri/marker?tab=readme-ov-file). In the (near) future, we plan to use the pdfs directly.

- Create a python virtual environment (tested on Python 3.12.9) and install the requirements.txt
- Prepare a folder `md_out` containing the markdown of your articles.
- **Make sure you have a valid openai api key in the .env file**
- Activate the virtual environment and run 

```bash
python create_citations_db.py
```


A file called `citations.db` will be created (or updated) in the same directory, which is a SQLite database. You can inspect it e.g. within a notebook with the following:

```python
import sqlite3
import pandas as pd

db_filename = "citations.db"

conn = sqlite3.connect(db_filename)
df = pd.read_sql_query("SELECT * FROM citations", conn)
conn.close()

df.style
```


See a [sample output](https://github.com/leonardocerliani/Jorge4PubMed/tree/main?tab=readme-ov-file#sample-output) at the bottom of this page

You can subsequently add more articles in the `md_out` directory. Only the newly added articles will be processed.

**NB**: At the moment, Jorge does not make a distinction between same articles with different filenames.

If you prefer to inspect all the steps of the processing, you can use the corresponding colab notebook.


## Performance
When tested on a corpus of 1729 papers, Jorge correctly retrieved 83% of the PMID. We expect to increase this figure in the (near) future.

## Applications
Once you have the PMID of each of your source papers, a few options open up, for instance:
- First, you can provide the citation in your RAG-generated response (as shown above)
- You can use the PMID to retrieve from the PubMed api all the papers citing that source
- You can specifically retrieve the paper abstract, which can be used e.g. to further validate the RAG response or provided to the user in the RAG app.

## Contributions
Contributions are welcome! Please feel free to submit a Pull Request.

## License 
This project is licensed under the MIT License.

## Disclaimer
This tool is for research purposes only. Please respect PubMed's terms of service and use this tool responsibly.


## Sample output

<details>
<summary>Click to expand the table</summary>

| ID | title                                                                                                                                                                                                                                                |  PMID   | filename                                | full_citation                                                                                                                                                                                                       | weblink                                | Pubmed_title                                                                                                                                                                              |
|----|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------:|:--------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | RISC Assembly Defects in the Drosophila RNAi Mutant Armitage                                                                                                                                                                                         | 15035985| 1009Tomari2004.md                     | Tomari (2004). _RISC assembly defects in the Drosophila RNAi mutant armitage._ Cell                                                                                                                               | https://pubmed.ncbi.nlm.nih.gov/15035985 | RISC assembly defects in the Drosophila RNAi mutant armitage.                                                                                                                             |
|  1 | Regulation of Constitutive GPR3 Signaling and Surface Localization by GRK2 and b-arrestin-2 Overexpression in HEK293 Cells                                                                                                                           | 23826079| 11Lowther2013.md                      | Lowther (2013). _Regulation of Constitutive GPR3 Signaling and Surface Localization by GRK2 and β-arrestin-2 Overexpression in HEK293 Cells._ PloS one                                                          | https://pubmed.ncbi.nlm.nih.gov/23826079 | Regulation of Constitutive GPR3 Signaling and Surface Localization by GRK2 and β-arrestin-2 Overexpression in HEK293 Cells.                                                                  |
|  2 | A Role for Calcium Release-Activated Current CRAC in Cholinergic Modulation of Electrical Activity in Pancreatic 3-Cells                                                                                                                             |  7647236| 3953Bertram1995.md                    | Bertram (1995). _A role for calcium release-activated current (CRAC) in cholinergic modulation of electrical activity in pancreatic beta-cells._ Biophysical journal                                              | https://pubmed.ncbi.nlm.nih.gov/7647236  | A role for calcium release-activated current (CRAC) in cholinergic modulation of electrical activity in pancreatic beta-cells.                                                              |

</details>