# Filename for the db of citations db
# (contains filename, title, PMID, full_citation, weblink)
db_filename = "citations.db"
logfile = "log_citation_db_creation.log"

# Set your email here to use NCBI Entrez API
email_for_entrez_service = "my_email@gmail.com"
limit_esearch_per_second = 3 # Use n > 3 ONLY if you have a PubMed api key

# Location of the markdown files extracted from the papers' pdf
md_folder = 'md_out'
n_lines = 20


# Batch size for extracting title using gpt-4o
llm_extract_titles_batch_size = 10


# ---------------- MODIFY THE CODE BELOW AT YOUR OWN RISK ----------------------



import time
import os
import sqlite3
import pandas as pd
from pprint import pprint
import concurrent.futures
from IPython.display import display

from Bio import Entrez
Entrez.email = email_for_entrez_service

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI()

import logging
logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s - %(message)s')

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



# Cosine similarity between to sentences
model = SentenceTransformer('all-MiniLM-L6-v2')

def batch_sentence_similarity(list1, list2):
    assert len(list1) == len(list2), "Lists must be of equal length"
    embeddings1 = model.encode(list1, show_progress_bar=False)
    embeddings2 = model.encode(list2, show_progress_bar=False)
    similarities = np.array([
        cosine_similarity([emb1], [emb2])[0][0]
        for emb1, emb2 in zip(embeddings1, embeddings2)
    ])
    return similarities


# Create citations.db if not already existing
def create_citations_db_if_not_existing(conn):
    cursor = conn.cursor()
    print(f"Database '{db_filename}' loaded or created successfully.")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS citations (
        title TEXT,
        PMID TEXT,
        filename TEXT,
        full_citation TEXT,
        weblink TEXT,
        Pubmed_title TEXT
    )
    """)
    pprint(cursor.execute("PRAGMA table_info(citations)").fetchall())


# Get new papers filename
def get_new_papers_filename(md_folder, n_lines, conn):
    new_papers = []
    cursor = conn.cursor()

    cursor.execute("SELECT filename FROM citations")
    existing_filenames = [row[0] for row in cursor.fetchall()]

    # Log duplicated filenames found in md_folder that already exist in DB
    for filename in os.listdir(md_folder):
        if filename.endswith('.md'):
            if filename in existing_filenames:
                logging.warning(f"Duplicate filename detected: '{filename}' already exists in the database.")

    print(f"Scanning folder '{md_folder}' for new markdown files...")

    for filename in os.listdir(md_folder):
        if filename.endswith('.md'):
            path = os.path.join(md_folder, filename)

            with open(path, 'r', encoding='utf-8') as f:
                lines = []
                for _ in range(n_lines):
                    try:
                        lines.append(next(f).strip())
                    except StopIteration:
                        break
                multiline_string = "\n".join(lines)

            if filename not in existing_filenames:
                new_papers.append({
                    "title": None,
                    "PMID": None,
                    "filename": filename,
                    "first_lines": multiline_string
                })

    print(f"Found {len(new_papers)} new markdown files to process.")
    return new_papers



# Extract titles from the md using gpt-4.1-nano
def extract_one_title(content):
    extract_title_parsing_instructions = """
    The following are the first lines of a scientific paper in markdown format.
    You task is to extract the title and return **ONLY** the title and nothing else.

    - Ignore any markdown formatting, headings, or special characters (e.g., `#`, `**`, etc.).
    - Try to correct spelling mistakes for common english words.
    - Do not correct anything in technical terms.

    It might be helpful to know that:
    - the very first line can be a collection heading (e.g. "Opinions", "Original Research", "Review paper", "Sort Communication") instead of the title.
    - the title is usually just before the author's list
    """

    response = client.responses.create(
        model='gpt-4.1-nano',
        input=content,
        instructions=extract_title_parsing_instructions,
        temperature=0.3
    )
    return response.output_text


def llm_extract_titles(new_papers, batch_size=llm_extract_titles_batch_size):
    total = len(new_papers)
    print(f"Extracting titles from {total} papers using GPT-4.1-nano in batches of {batch_size}...")

    for i in range(0, total, batch_size):
        batch = new_papers[i:i + batch_size]
        contents = [paper["first_lines"] for paper in batch]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(extract_one_title, contents))

        for paper, title in zip(batch, results):
            paper["title"] = title
            del paper["first_lines"]

        print(f"Processed titles for papers {i+1} to {min(i+batch_size, total)} / {total}")


def log_duplicate_titles(new_papers, conn):
    cursor = conn.cursor()
    cursor.execute("SELECT title, filename FROM citations WHERE title IS NOT NULL")
    existing_records = cursor.fetchall()

    existing_titles = {}
    for title, filename in existing_records:
        if title in existing_titles:
            existing_titles[title].append(filename)
        else:
            existing_titles[title] = [filename]

    for paper in new_papers:
        title = paper['title']
        if title in existing_titles:
            existing_files = ', '.join(existing_titles[title])
            logging.warning(f"Duplicate title found: '{title}' from file '{paper['filename']}' matches existing files: {existing_files}")




# Initial search of PMID on Pubmed esearch
def fetch_PMIDs_with_rate_limit(new_papers, proximity=2, limit_esearch_per_second=3):
    total = len(new_papers)
    processed = 0

    for paper in new_papers:
        title = paper["title"]
        try:
            handle = Entrez.esearch(db="pubmed", term=f'"{title}"[Title:~{proximity}]', retmax=1)
            record = Entrez.read(handle)
            handle.close()
            id_list = record.get("IdList")
            paper["PMID"] = id_list[0] if id_list else ''
        except Exception as e:
            logging.error(f"Error fetching PMID for title '{title}': {e}")
            paper["PMID"] = ''

        processed += 1
        print(f"Fetched PMID for {processed}/{total} papers", end="\r", flush=True)

        if processed % limit_esearch_per_second == 0:
            time.sleep(1)

    print()
    


# Get the titles of the identified PMIDs using efetch
def get_pubmed_titles(new_papers, batch_size=100):
    papers_with_pmid = [paper for paper in new_papers if paper.get('PMID')]
    if not papers_with_pmid:
        print("No papers with PMIDs found for validation.")
        return

    print(f"Fetching Pubmed titles for {len(papers_with_pmid)} papers...")

    for i in range(0, len(papers_with_pmid), batch_size):
        batch = papers_with_pmid[i:i + batch_size]
        pmids_string = ",".join([p['PMID'] for p in batch])

        try:
            handle = Entrez.efetch(db="pubmed", id=pmids_string, retmode="xml")
            records = Entrez.read(handle)
            handle.close()

            for paper, article in zip(batch, records['PubmedArticle']):
                try:
                    fetched_title = article['MedlineCitation']['Article']['ArticleTitle']
                    paper['Pubmed_title'] = fetched_title  # Just store the title
                except Exception as e:
                    logging.error(f"Error fetching title for PMID {paper.get('PMID')}: {e}")
                    paper['Pubmed_title'] = ''

            print(f"Fetched Pubmed titles for papers {i+1} to {min(i+batch_size, len(papers_with_pmid))}")

        except Exception as e:
            logging.error(f"Entrez fetch error for batch starting at index {i}: {e}")
        
        print("Sleeping for 2 seconds to comply with Pubmed rate limit")
        time.sleep(2)



# Reject PMID where the similarity of title with Pubmed_title is < 0.94
def compare_titles_embeddings(new_papers,similarity_threshold):
    # Filter papers where Pubmed_title exists and is non-empty
    filtered = [p for p in new_papers if p.get('Pubmed_title')]

    if not filtered:
        print("No papers with 'Pubmed_title' to compare.")
        return pd.DataFrame()

    titles = [p['title'] for p in filtered]
    pubmed_titles = [p['Pubmed_title'] for p in filtered]
    pmids = [p['PMID'] for p in filtered]

    similarities = batch_sentence_similarity(titles, pubmed_titles)

    # Update original papers based on similarity
    for paper, sim in zip(filtered, similarities):
      if sim < similarity_threshold:
          paper['PMID'] = ''

    df = pd.DataFrame({
        'title': [p['title'] for p in filtered],
        'Pubmed_title': [p.get('Pubmed_title', '') for p in filtered],
        'cosine_similarity': similarities,
        'PMID': [p['PMID'] for p in filtered]
    })

    return df


# Deep search using window_of_words and proximity
def word_window_proximity_search(new_papers, window_of_words_size=5, proximity_deep_search=0, limit_esearch_per_second = 3):
    print(f"Starting word window proximity search for missing PMIDs (window size = {window_of_words_size}, proximity = {proximity_deep_search})...")

    for idx, paper in enumerate(new_papers, 1):
        if paper.get("PMID", '') == '':
            print(f"[{idx}/{len(new_papers)}] Searching PMID for: {paper['title']}")

            words = paper['title'].split()
            if len(words) < window_of_words_size:
                continue

            pmid_counts = {}

            for i in range(len(words) - window_of_words_size + 1):
                window_words = words[i:i+window_of_words_size]
                query = f"\"{' '.join(window_words)}\"[Title:~{proximity_deep_search}]"

                try:
                    handle = Entrez.esearch(db="pubmed", term=query, retmax=1)
                    record = Entrez.read(handle)
                    handle.close()
                except Exception as e:
                    print(f"Error during Entrez search for query '{query}': {e}")
                    continue

                id_list = record.get("IdList", [])
                if id_list:
                    for pmid in id_list:
                        pmid_counts[pmid] = pmid_counts.get(pmid, 0) + 1
                    print(f"Attempt {i+1}: Found PMIDs {id_list}")

                # Sleep after every n requests to keep rate limit
                # Use n > 3 ONLY if you have a PubMed api key
                if (i + 1) % limit_esearch_per_second == 0:
                    time.sleep(1)

            if pmid_counts:
                most_common_pmid = max(pmid_counts, key=pmid_counts.get)
                print(f"Most frequent PMID: {most_common_pmid}")
                paper["PMID"] = most_common_pmid

            print("Sleeping for 2 seconds to comply with Pubmed rate limit")
            time.sleep(2)

    print("Completed word window proximity search.")



# Compare llm-retrieved title with Pubmed_title with window_of_words similarity after deep title search. 
# Remove PMID if similarity is < 0.94
def window_compare_titles_embeddings(new_papers, window_size=5, similarity_threshold=0.94):

    def get_windows(words, size):
        return [words[i:i+size] for i in range(len(words) - size + 1)] if len(words) >= size else []

    filtered = [p for p in new_papers if p.get('Pubmed_title')]

    if not filtered:
        print("No papers with 'Pubmed_title' to compare.")
        return pd.DataFrame()

    max_sims = []
    keep_flags = []
    for paper in filtered:
        title_words = paper['title'].split()
        pubmed_words = paper['Pubmed_title'].split()

        if len(title_words) < window_size or len(pubmed_words) < window_size:
            max_sim = 0
            # paper['PMID'] = ''  # commented out for testing
            keep = False
        else:
            title_windows = get_windows(title_words, window_size)
            pubmed_windows = get_windows(pubmed_words, window_size)

            similarities = []
            n = min(len(title_windows), len(pubmed_windows))
            for i in range(n):
                title_segment = " ".join(title_windows[i])
                pubmed_segment = " ".join(pubmed_windows[i])
                sim = batch_sentence_similarity([title_segment], [pubmed_segment])[0]
                similarities.append(sim)

            max_sim = max(similarities) if similarities else 0

            # Comment this 'if' for testing with different similarity thresholds
            if max_sim < similarity_threshold:
                paper['PMID'] = ''

            keep = max_sim >= similarity_threshold

        max_sims.append(max_sim)
        keep_flags.append(keep)

    df = pd.DataFrame({
        'title': [p['title'] for p in filtered],
        'Pubmed_title': [p.get('Pubmed_title', '') for p in filtered],
        'max_window_cosine_similarity': max_sims,
        'PMID': [p['PMID'] for p in filtered],
        'kept_based_on_threshold': keep_flags
    })

    return df



# Add full citation
def add_full_citation(new_papers, batch_size=100):
    papers_with_pmid = [paper for paper in new_papers if paper.get('PMID')]
    if not papers_with_pmid:
        print("No papers with PMIDs found for citation fetching.")
        return

    print(f"Fetching full citations for {len(papers_with_pmid)} papers...")

    for i in range(0, len(papers_with_pmid), batch_size):
        batch = papers_with_pmid[i:i + batch_size]
        pmids_string = ",".join([p['PMID'] for p in batch])

        try:
            handle = Entrez.efetch(db="pubmed", id=pmids_string, retmode="xml")
            records = Entrez.read(handle)
            handle.close()

            for paper, article in zip(batch, records['PubmedArticle']):
                try:
                    article_data = article['MedlineCitation']['Article']
                    journal_info = article_data.get('Journal', {})
                    pub_date = journal_info.get('JournalIssue', {}).get('PubDate', {})

                    title = article_data.get('ArticleTitle')
                    journal = journal_info.get('Title')
                    year = pub_date.get('Year')

                    authors = article_data.get('AuthorList', [])
                    first_author_surname = authors[0].get('LastName') if authors else None

                    full_citation = f'{first_author_surname} ({year}). _{title}_ {journal}'

                    paper['full_citation'] = full_citation
                    paper['weblink'] = f'https://pubmed.ncbi.nlm.nih.gov/{paper["PMID"]}'
                    paper['Pubmed_title'] = title
                except (KeyError, IndexError, TypeError) as e:
                    logging.error(f"Error processing article for PMID {paper['PMID']}: {e}")

            print(f"Processed citations for papers {i+1} to {min(i+batch_size, len(papers_with_pmid))}")

        except Exception as e:
            logging.error(f"Error during Entrez fetch for PMIDs batch starting at index {i}: {e}")



# Insert new papers in the citations.db (in batches of 100)
def insert_new_papers(new_papers, conn, batch_size=100):
    cursor = conn.cursor()
    total = len(new_papers)
    print(f"Ingesting {total} new papers into the database in batches of {batch_size}...")

    for idx, paper in enumerate(new_papers, 1):
        try:
            title = paper.get('title')
            PMID = paper.get('PMID') or None
            filename = paper.get('filename')
            full_citation = paper.get('full_citation')
            weblink = paper.get('weblink')
            Pubmed_title = paper.get('Pubmed_title')

            cursor.execute('''
                INSERT INTO citations (title, PMID, filename, full_citation, weblink, Pubmed_title)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (title, PMID, filename, full_citation, weblink, Pubmed_title))

            if idx % batch_size == 0 or idx == total:
                conn.commit()
                print(f"Committed batch up to paper {idx}/{total}")

        except Exception as e:
            logging.error(f"Failed to insert paper '{filename}': {e}")
            continue





def main():
    print("Starting citation ingestion process...\n")

    conn = None
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_filename)

        # Ensure citations table exists
        create_citations_db_if_not_existing(conn)

        # Extract new papers from markdown files not yet in the DB
        new_papers = get_new_papers_filename(md_folder, n_lines, conn)

        if not new_papers:
            print("No new markdown files found. Exiting.")
            return

        # Extract titles from the markdown files using an LLM
        llm_extract_titles(new_papers, batch_size=10)

        # Log any duplicate titles that already exist in the database
        log_duplicate_titles(new_papers, conn)


        # ---------- Initial PMID search with proximity = 2 --------------------
        proximity_initial_search = 2
        limit_esearch_per_second = 3 # Use n > 3 ONLY if you have a PubMed api key

        fetch_PMIDs_with_rate_limit(
            new_papers, 
            proximity = proximity_initial_search, 
            limit_esearch_per_second = limit_esearch_per_second
        )

        # Get the Pubmed_title for the found PMID
        get_pubmed_titles(new_papers)


        # Verify after INITIAL SEARCH that the similarity between llm-extracted
        # and efetch-ed Pubmed_title is > 0.94 otherwise remove PMID
        similarity_threshold_initial_search = 0.94

        df_initial_search = compare_titles_embeddings(
            new_papers,
            similarity_threshold=similarity_threshold_initial_search
        )



        # ---------- Deep search of PMID with strict proximity -----------------
        # Carry out the window_of_words search
        window_of_words_size = 5
        proximity_deep_search = 0
        limit_esearch_per_second = 3 # Use n > 3 ONLY if you have a PubMed api key

        word_window_proximity_search(
            new_papers, 
            window_of_words_size, 
            proximity_deep_search,
            limit_esearch_per_second
        )


        # add the Pubmed_title
        get_pubmed_titles(new_papers)


        # Verify after DEEP SEARCH that the similarity between llm-extracted
        # and efetch-ed Pubmed_title is > 0.94 otherwise remove PMID
        similarity_threshold_deep_search = 0.94

        df_deep_search = window_compare_titles_embeddings(
            new_papers, window_size=5, similarity_threshold=similarity_threshold_deep_search
        )


        # Fetch and store full citation details for validated PMIDs
        add_full_citation(new_papers)

        # Insert the newly processed papers into the citations database
        insert_new_papers(new_papers, conn)

        print("\nCitation ingestion process completed.")

        # Print summary of inserted records
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM citations")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM citations WHERE PMID != ''")
        with_pmid = cursor.fetchone()[0]

        without_pmid = total - with_pmid
        percentage_with = (with_pmid / total) * 100 if total else 0

        print(f"\nSummary of inserted records:")
        print(f"- Total records: {total}")
        print(f"- With PMID: {with_pmid}")
        print(f"- Without PMID: {without_pmid}")
        print(f"- Percentage with PMID: {percentage_with:.2f}%")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if conn:
            conn.close()


if __name__ == '__main__':
    main()
