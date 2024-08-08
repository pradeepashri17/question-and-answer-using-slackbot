from typing import Dict, List, Union, Tuple
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import json



def parse_pdf_pages(file_path: str) -> Dict[int, str]:
   
    page_dict: Dict[int, str] = {}

    for page_num, page_layout in enumerate(extract_pages(file_path), start=1):
        page_content = ""
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                page_content += element.get_text()


        # Append the NER results at the top of the page content
        page_content = f"Page Content:\n{page_content.strip()}"

        page_dict[page_num] = page_content

    return page_dict


def store_pages_in_faiss(pages: Dict[int, str]) -> Tuple[faiss.IndexFlatL2, TfidfVectorizer]:
    
    # Convert the page contents to a list of strings
    page_contents = list(pages.values())

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Vectorize the page contents
    page_vectors: np.ndarray = vectorizer.fit_transform(page_contents).toarray()

    # Create a FAISS index
    index = faiss.IndexFlatL2(page_vectors.shape[1])

    # Add the page vectors to the FAISS index
    index.add(page_vectors)

    return index, vectorizer


def search_pages(query: str, index: faiss.IndexFlatL2, vectorizer: TfidfVectorizer, pages: Dict[int, str], top_k: int = 5) -> List[Dict[str, Union[int, str]]]:
   
    # Vectorize the query using the same vectorizer
    query_vector: np.ndarray = vectorizer.transform([query]).toarray()

    # Perform the similarity search
    distances, indices = index.search(query_vector, top_k)

    # Create a list of dictionaries representing the top similar pages
    results = []
    for idx in indices[0]:
        page_number = list(pages.keys())[idx]
        page_content = pages[page_number]
        results.append({'page_number': page_number, 'content': page_content})

    return results


def main() -> None:
    pdf_file_path = 'handbook.pdf'
    parsed_pages = parse_pdf_pages(pdf_file_path)

    print("PRINTING PAGE DICTIONARY")
    print(parsed_pages)
    print("---")

    # Store the parsed pages in a FAISS index and get the vectorizer
    index, vectorizer = store_pages_in_faiss(parsed_pages)

    # Perform a similarity search
    query = "vacation policy"
    top_similar_pages = search_pages(query, index, vectorizer, parsed_pages)

    print(f"TOP SIMILAR PAGES FOR QUERY: '{query}'")
    for page in top_similar_pages[:-2]:
        print(f"Page {page['page_number']}:")
        print(page['content'])
        print("---")


if __name__ == "__main__":
    main()
