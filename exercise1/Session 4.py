#%pip install langchain_ollama
#%pip install langchain_text_splitters
#%pip install langchain_community
#%pip install langchain_huggingface
#%pip install pypdf
#%pip install rank_bm25
#%% EXERCISE 1: INTRODUCTION TO RAG
# LLM Only vs LLM + Retrieved Evidence
# Business Case: Pre-Audit Supporting Document Assistant

from langchain_ollama import ChatOllama
# ----------------------------------------------------------
# 1. LOCAL LLM THROUGH LANGCHAIN
# ----------------------------------------------------------

llm = ChatOllama(
    model="gemma3:4b",
    temperature=0
)


# ----------------------------------------------------------
# 2. USER QUESTION
# ----------------------------------------------------------

question = """
I attended a local conference and paid the travel
expenses myself.

What supporting documents should I submit
for reimbursement?
"""


# ----------------------------------------------------------
# 3. LLM ONLY - STREAMING RESPONSE
# ----------------------------------------------------------

print("USER QUESTION:")
print(question)

print("\n" + "=" * 75)
print("LLM ONLY:")
print("-" * 75)

for chunk in llm.stream(
    question
):

    print(
        chunk.content,
        end="",
        flush=True
    )


# ----------------------------------------------------------
# 4. SIMULATED RETRIEVED EVIDENCE
# ----------------------------------------------------------
# In Exercise 1, the evidence is manually supplied.
# In later exercises, the evidence will be retrieved
# automatically from the PDF.

retrieved_evidence = """
[C01] Travel Reimbursement

For reimbursement of local or foreign travel,
submit the Disbursement Voucher (DV) and approved
Obligation Request (ObR) or Budget Utilization
Request (BUR).

[C02] Travel Supporting Documents

Supporting documents include:

- Summary of expenses
- Invitation for conferences, seminars, or similar activities
- Travel Order approved by the Chancellor
- Actual Itinerary of Travel
- Certificate of Travel Completed
- Certificate of Appearance
- Original boarding pass and terminal fee stub
- Invoice for airfare or registration fee

[C03] Conference Requirement

For seminar, workshop, conference, and strategic
planning activities, a Narrative Report is also required.
"""


# ----------------------------------------------------------
# 5. AUGMENTED PROMPT
# ----------------------------------------------------------

augmented_question = f"""
You are a pre-audit assistant for a University
Accounting Office.

Answer the user's question using only the approved
checklist evidence provided below.

Do not invent requirements.
Do not add documents that are not stated in the evidence.

If the evidence does not contain enough information,
say that the checklist evidence is insufficient.

Cite the supporting chunk IDs such as [C01], [C02],
and [C03] when appropriate.

APPROVED CHECKLIST EVIDENCE:

{retrieved_evidence}

USER QUESTION:

{question}
"""


# ----------------------------------------------------------
# 6. RAG - STREAMING RESPONSE
# ----------------------------------------------------------

print("\n\n" + "=" * 75)
print("LLM + RETRIEVED EVIDENCE:")
print("-" * 75)

for chunk in llm.stream(
    augmented_question
):

    print(
        chunk.content,
        end="",
        flush=True
    )

print("\n")

#%% EXERCISE 2: PREPARING AND STORING A PDF FOR RAG
# PDF → Page-Aware Chunking → SQLite Knowledge Store

import os
import sqlite3
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ----------------------------------------------------------
# 1. FILES IN THE WORKING DIRECTORY
# ----------------------------------------------------------

pdf_file = "PreAudit Checklist for Common Transactions.pdf"

database_file = "rag_knowledge.db"


# ----------------------------------------------------------
# 2. VERIFY REQUIRED FILES
# ----------------------------------------------------------

if not os.path.exists(pdf_file):
    raise FileNotFoundError(
        f"PDF file not found: {pdf_file}"
    )

if not os.path.exists(database_file):
    raise FileNotFoundError(
        f"SQLite database not found: {database_file}"
    )


# ----------------------------------------------------------
# 3. LOAD THE PDF
# ----------------------------------------------------------

loader = PyPDFLoader(
    pdf_file
)

pages = loader.load()


# ----------------------------------------------------------
# 4. REMOVE EMPTY PAGES
# ----------------------------------------------------------

pages = [
    page
    for page in pages
    if page.page_content.strip()
]


# ----------------------------------------------------------
# 5. ADD TRACEABILITY METADATA
# ----------------------------------------------------------
# PyPDFLoader normally stores page numbers beginning at 0.
# Add a reader-friendly page number beginning at 1.

for page in pages:

    page.metadata["page_number"] = (
        page.metadata.get("page", 0) + 1
    )

    page.metadata["source_name"] = (
        pdf_file
    )


# ----------------------------------------------------------
# 6. CREATE TEXT SPLITTER
# ----------------------------------------------------------
# Each chunk contains up to approximately 900 characters.
# The 150-character overlap helps preserve information
# that may cross chunk boundaries.

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)


# ----------------------------------------------------------
# 7. SPLIT PDF INTO RETRIEVAL-READY CHUNKS
# ----------------------------------------------------------

chunks = text_splitter.split_documents(
    pages
)


# ----------------------------------------------------------
# 8. ASSIGN UNIQUE CHUNK IDS
# ----------------------------------------------------------

for i, chunk in enumerate(
    chunks,
    start=1
):

    chunk.metadata["chunk_id"] = (
        f"C{i:03d}"
    )


# ----------------------------------------------------------
# 9. CONNECT TO EXISTING SQLITE DATABASE
# ----------------------------------------------------------

connection = sqlite3.connect(
    database_file
)

cursor = connection.cursor()


# ----------------------------------------------------------
# 10. VERIFY THE CHUNKS TABLE EXISTS
# ----------------------------------------------------------

table_exists = cursor.execute(
    """
    SELECT name
    FROM sqlite_master
    WHERE type='table'
      AND name='chunks'
    """
).fetchone()

if table_exists is None:

    connection.close()

    raise RuntimeError(
        "The chunks table was not found in rag_knowledge.db."
    )


# ----------------------------------------------------------
# 11. CLEAR PREVIOUS CHUNKS
# ----------------------------------------------------------
# This lets participants safely rerun the exercise without
# creating duplicate records.

cursor.execute(
    """
    DELETE FROM chunks
    """
)


# ----------------------------------------------------------
# 12. PREPARE CHUNKS FOR SQLITE
# ----------------------------------------------------------

rows_to_insert = [
    (
        chunk.metadata["chunk_id"],
        chunk.metadata["page_number"],
        chunk.metadata["source_name"],
        chunk.page_content
    )
    for chunk in chunks
]


# ----------------------------------------------------------
# 13. STORE CHUNKS IN SQLITE
# ----------------------------------------------------------

cursor.executemany(
    """
    INSERT INTO chunks (
        chunk_id,
        page_number,
        source,
        chunk_text
    )
    VALUES (?, ?, ?, ?)
    """,
    rows_to_insert
)

connection.commit()


# ----------------------------------------------------------
# 14. READ STORED CHUNKS BACK FROM SQLITE
# ----------------------------------------------------------

chunk_summary = pd.read_sql_query(
    """
    SELECT
        chunk_id,
        page_number AS page,
        source,
        LENGTH(chunk_text) AS characters,
        REPLACE(
            SUBSTR(chunk_text, 1, 120),
            CHAR(10),
            ' '
        ) AS preview
    FROM chunks
    ORDER BY chunk_id
    """,
    connection
)


# ----------------------------------------------------------
# 15. VERIFY NUMBER OF STORED CHUNKS
# ----------------------------------------------------------

stored_count = cursor.execute(
    """
    SELECT COUNT(*)
    FROM chunks
    """
).fetchone()[0]


# ----------------------------------------------------------
# 16. DISPLAY INGESTION SUMMARY
# ----------------------------------------------------------

print("PDF FILE:")
print(pdf_file)

print("\nSQLITE DATABASE:")
print(database_file)

print("\nNUMBER OF PDF PAGES:")
print(len(pages))

print("\nNUMBER OF CHUNKS CREATED:")
print(len(chunks))

print("\nNUMBER OF CHUNKS STORED IN SQLITE:")
print(stored_count)

print("\nCHUNK SUMMARY:")
print(
    chunk_summary.to_string(
        index=False
    )
)


# ----------------------------------------------------------
# 17. INSPECT FIRST THREE CHUNKS FROM SQLITE
# ----------------------------------------------------------

print("\n" + "=" * 75)
print("FIRST THREE CHUNKS RETRIEVED FROM SQLITE")

stored_rows = cursor.execute(
    """
    SELECT
        chunk_id,
        page_number,
        source,
        chunk_text
    FROM chunks
    ORDER BY chunk_id
    LIMIT 3
    """
).fetchall()

for row in stored_rows:

    print("\n" + "-" * 70)

    print(
        "CHUNK ID:",
        row[0]
    )

    print(
        "PAGE:",
        row[1]
    )

    print(
        "SOURCE:",
        row[2]
    )

    print("\nTEXT:")
    print(
        row[3]
    )


# ----------------------------------------------------------
# 18. CLOSE DATABASE CONNECTION
# ----------------------------------------------------------

connection.close()

print("\n" + "=" * 75)
print("PDF INGESTION COMPLETED.")
print(
    "The chunks are now stored persistently "
    "in rag_knowledge.db."
)
#%% EXERCISE 3: KEYWORD SEARCH WITH BM25
# SQLite Helper → LangChain Documents → BM25 Scoring → Ranked Chunks

import os
import re
import sqlite3

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


# ==========================================================
# HELPER FUNCTION: LOAD CHUNKS FROM SQLITE
# ==========================================================

def load_chunks_from_sqlite(
    database_file="rag_knowledge.db"
):

    # ------------------------------------------------------
    # 1. VERIFY DATABASE EXISTS
    # ------------------------------------------------------

    if not os.path.exists(database_file):

        raise FileNotFoundError(
            f"SQLite database not found: {database_file}\n"
            "Run Exercise 2 first to populate the knowledge base."
        )


    # ------------------------------------------------------
    # 2. CONNECT TO SQLITE
    # ------------------------------------------------------

    connection = sqlite3.connect(
        database_file
    )

    cursor = connection.cursor()


    # ------------------------------------------------------
    # 3. VERIFY CHUNKS TABLE EXISTS
    # ------------------------------------------------------

    table_exists = cursor.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
          AND name='chunks'
        """
    ).fetchone()


    if table_exists is None:

        connection.close()

        raise RuntimeError(
            "The chunks table was not found "
            "in the SQLite database."
        )


    # ------------------------------------------------------
    # 4. LOAD STORED CHUNKS
    # ------------------------------------------------------

    rows = cursor.execute(
        """
        SELECT
            chunk_id,
            page_number,
            source,
            chunk_text
        FROM chunks
        ORDER BY chunk_id
        """
    ).fetchall()

    connection.close()


    # ------------------------------------------------------
    # 5. VERIFY KNOWLEDGE BASE IS NOT EMPTY
    # ------------------------------------------------------

    if len(rows) == 0:

        raise RuntimeError(
            "No chunks are stored in the database.\n"
            "Run Exercise 2 first."
        )


    # ------------------------------------------------------
    # 6. CONVERT ROWS TO LANGCHAIN DOCUMENTS
    # ------------------------------------------------------

    chunks = [
        Document(
            page_content=row[3],
            metadata={
                "chunk_id": row[0],
                "page_number": row[1],
                "source_name": row[2]
            }
        )
        for row in rows
    ]


    return chunks


# ==========================================================
# EXERCISE CORE
# ==========================================================


# ----------------------------------------------------------
# 1. LOAD CHUNKS FROM SQLITE
# ----------------------------------------------------------

chunks = load_chunks_from_sqlite(
    "rag_knowledge.db"
)

print(
    "Chunks loaded from SQLite:",
    len(chunks)
)


# ----------------------------------------------------------
# 2. SIMPLE TOKENIZER FOR BM25
# ----------------------------------------------------------
# BM25 performs lexical search using words/tokens.
# Lowercasing allows Travel and travel to be treated
# as the same term.

def tokenize(text):

    return re.findall(
        r"\b[\w.-]+\b",
        text.lower()
    )


# ----------------------------------------------------------
# 3. BUILD LANGCHAIN BM25 RETRIEVER
# ----------------------------------------------------------
# The chunks were already persisted in SQLite.
# BM25 now builds an in-memory keyword index over them.

retriever = BM25Retriever.from_documents(
    chunks,
    preprocess_func=tokenize
)

retriever.k = 5


# ----------------------------------------------------------
# 4. TEST QUERIES
# ----------------------------------------------------------

queries = [

    # Exact and distinctive terms from the checklist
    "Certificate of Appearance travel reimbursement",

    # Keyword-rich query from another transaction type
    "food purchases attendance agenda",

    # Conversational wording
    # This illustrates a limitation of lexical retrieval.
    "What documents do I need after returning from an official trip?"
]


# ----------------------------------------------------------
# 5. RUN BM25 SEARCH
# ----------------------------------------------------------

for query in queries:


    # ------------------------------------------------------
    # A. TOKENIZE QUERY
    # ------------------------------------------------------

    query_tokens = tokenize(
        query
    )


    # ------------------------------------------------------
    # B. CALCULATE BM25 SCORES
    # ------------------------------------------------------

    scores = (
        retriever.vectorizer.get_scores(
            query_tokens
        )
    )


    # ------------------------------------------------------
    # C. SORT CHUNKS BY SCORE
    # ------------------------------------------------------

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    top_indices = ranked_indices[
        :retriever.k
    ]


    # ------------------------------------------------------
    # D. DISPLAY QUERY
    # ------------------------------------------------------

    print("\n" + "=" * 75)

    print("QUERY:")
    print(query)

    print("\nQUERY TOKENS:")
    print(query_tokens)

    print("\nBM25 RANKING:")


    # ------------------------------------------------------
    # E. DISPLAY TOP-K RESULTS
    # ------------------------------------------------------

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        doc = retriever.docs[index]

        score = float(
            scores[index]
        )


        # --------------------------------------------------
        # FIND QUERY TERMS THAT APPEAR IN THE CHUNK
        # --------------------------------------------------

        document_tokens = set(
            tokenize(
                doc.page_content
            )
        )

        matched_terms = sorted(
            set(query_tokens)
            & document_tokens
        )


        # --------------------------------------------------
        # DISPLAY RESULT
        # --------------------------------------------------

        print("\n" + "-" * 70)

        print(
            f'Rank {rank}: '
            f'[{doc.metadata["chunk_id"]}] '
            f'Page {doc.metadata["page_number"]}'
        )

        print(
            "BM25 Score:",
            round(
                score,
                4
            )
        )

        print(
            "Matched Query Terms:",
            matched_terms
        )

        print(
            "Source:",
            doc.metadata["source_name"]
        )

        print("\nTEXT:")

        print(
            doc.page_content
        )
#%% EXERCISE 4: SEMANTIC SIMILARITY AND RANKING
# SQLite Helper → SBERT Embeddings → Semantic Search
# → LangChain Vector Retrieval

import os
import sqlite3
import pandas as pd

from sentence_transformers import SentenceTransformer

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# ----------------------------------------------------------
# 1. LOAD CHUNKS FROM SQLITE
# ----------------------------------------------------------

chunks = load_chunks_from_sqlite(
    "rag_knowledge.db"
)

print(
    "Chunks loaded from SQLite:",
    len(chunks)
)


# ----------------------------------------------------------
# 2. TEST QUERIES
# ----------------------------------------------------------
# These queries use conversational wording rather than
# exact phrases found in the checklist.

queries = [

    (
        "I just returned from an official trip and "
        "need to claim my expenses. "
        "What paperwork should I prepare?"
    ),

    (
        "I bought meals for an official meeting. "
        "What supporting documents are required?"
    )
]


# ----------------------------------------------------------
# 3. LOAD SBERT MODEL DIRECTLY
# ----------------------------------------------------------

model_name = (
    "sentence-transformers/"
    "multi-qa-MiniLM-L6-cos-v1"
)

print("\nLoading SBERT model...")

sbert_model = SentenceTransformer(
    model_name
)


# ----------------------------------------------------------
# 4. CREATE EMBEDDINGS FOR STORED CHUNKS
# ----------------------------------------------------------

chunk_texts = [
    chunk.page_content
    for chunk in chunks
]

chunk_embeddings = sbert_model.encode(
    chunk_texts,
    normalize_embeddings=True,
    convert_to_numpy=True
)

print(
    "Chunk embeddings created:",
    chunk_embeddings.shape
)


# ----------------------------------------------------------
# 5. DIRECT SBERT SEMANTIC SEARCH
# ----------------------------------------------------------

for query in queries:


    # ------------------------------------------------------
    # A. CREATE QUERY EMBEDDING
    # ------------------------------------------------------

    query_embedding = sbert_model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True
    )


    # ------------------------------------------------------
    # B. CALCULATE COSINE SIMILARITY
    # ------------------------------------------------------
    # Because the query and chunk embeddings are normalized,
    # their dot product equals cosine similarity.

    similarities = (
        chunk_embeddings
        @ query_embedding
    )


    # ------------------------------------------------------
    # C. CREATE RANKING TABLE
    # ------------------------------------------------------

    results = pd.DataFrame({

        "Chunk": [
            chunk.metadata["chunk_id"]
            for chunk in chunks
        ],

        "Page": [
            chunk.metadata["page_number"]
            for chunk in chunks
        ],

        "Similarity": similarities,

        "Preview": [
            chunk.page_content[:180]
            .replace("\n", " ")
            for chunk in chunks
        ]
    })


    # ------------------------------------------------------
    # D. SORT BY SEMANTIC SIMILARITY
    # ------------------------------------------------------

    results = results.sort_values(
        "Similarity",
        ascending=False
    ).reset_index(
        drop=True
    )

    results["Rank"] = (
        results.index + 1
    )


    # ------------------------------------------------------
    # E. KEEP TOP 5 RESULTS
    # ------------------------------------------------------

    top_results = results.head(5)


    # ------------------------------------------------------
    # F. DISPLAY RESULTS
    # ------------------------------------------------------

    print("\n" + "=" * 75)

    print("QUERY:")
    print(query)

    print("\nDIRECT SBERT SEMANTIC SEARCH:")

    print(
        top_results[
            [
                "Rank",
                "Chunk",
                "Page",
                "Similarity",
                "Preview"
            ]
        ].to_string(
            index=False,
            formatters={
                "Similarity": (
                    lambda x: f"{x:.4f}"
                )
            }
        )
    )


# ----------------------------------------------------------
# 6. USE THE SAME SBERT MODEL THROUGH LANGCHAIN
# ----------------------------------------------------------
# LangChain provides a wrapper around Sentence Transformers
# that can be connected directly to a vector store.

print("\n" + "=" * 75)
print("CREATING LANGCHAIN EMBEDDING MODEL")

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs={
        "normalize_embeddings": True
    }
)


# ----------------------------------------------------------
# 7. CREATE LANGCHAIN VECTOR STORE
# ----------------------------------------------------------
# SQLite stores the chunks persistently.
#
# The embeddings created here are stored only in memory
# for this classroom exercise.

vector_store = InMemoryVectorStore(
    embedding=embeddings
)

vector_store.add_documents(
    chunks
)

print(
    "Chunks added to LangChain vector store:",
    len(chunks)
)


# ----------------------------------------------------------
# 8. LANGCHAIN SEMANTIC SEARCH
# ----------------------------------------------------------

print("\n" + "=" * 75)
print("LANGCHAIN VECTOR STORE SEARCH")


for query in queries:

    results = (
        vector_store.similarity_search_with_score(
            query,
            k=5
        )
    )

    print("\n" + "=" * 75)

    print("QUERY:")
    print(query)

    print("\nLANGCHAIN SEMANTIC RANKING:")


    # ------------------------------------------------------
    # 9. DISPLAY TOP-K LANGCHAIN RESULTS
    # ------------------------------------------------------

    for rank, (doc, score) in enumerate(
        results,
        start=1
    ):

        print("\n" + "-" * 70)

        print(
            f'Rank {rank}: '
            f'[{doc.metadata["chunk_id"]}] '
            f'Page {doc.metadata["page_number"]}'
        )

        print(
            "Similarity:",
            round(
                float(score),
                4
            )
        )

        print(
            "Source:",
            doc.metadata["source_name"]
        )

        print("\nTEXT PREVIEW:")

        print(
            doc.page_content[:500]
        )

#%% EXERCISE 5: REUSABLE HYBRID RAG FUNCTION
# SQLite → BM25 + SBERT → RRF → Neighbor Expansion
# → Augmented Prompt → Gemma


import os
import re
import sqlite3
import numpy as np

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# ==========================================================
# ONE-TIME RAG SETUP
# ==========================================================

def expand_neighbor_chunks(
    chunks,
    top_indices,
    neighbor_window=1
):

    expanded_indices = []

    for index in top_indices:

        anchor_source = (
            chunks[index]
            .metadata["source_name"]
        )

        start_index = max(
            0,
            index - neighbor_window
        )

        end_index = min(
            len(chunks) - 1,
            index + neighbor_window
        )

        for neighbor_index in range(
            start_index,
            end_index + 1
        ):

            same_source = (
                chunks[neighbor_index]
                .metadata["source_name"]
                == anchor_source
            )

            if (
                same_source
                and neighbor_index
                not in expanded_indices
            ):

                expanded_indices.append(
                    neighbor_index
                )

    expanded_docs = [
        chunks[index]
        for index in expanded_indices
    ]

    return expanded_docs, expanded_indices



# ----------------------------------------------------------
# 1. LOAD KNOWLEDGE BASE
# ----------------------------------------------------------

chunks = load_chunks_from_sqlite(
    "rag_knowledge.db"
)

print(
    "Chunks loaded from SQLite:",
    len(chunks)
)


# ----------------------------------------------------------
# 2. BUILD BM25 RETRIEVER
# ----------------------------------------------------------

bm25_retriever = BM25Retriever.from_documents(
    chunks,
    preprocess_func=tokenize
)


# ----------------------------------------------------------
# 3. LOAD SBERT EMBEDDING MODEL
# ----------------------------------------------------------

model_name = (
    "sentence-transformers/"
    "multi-qa-MiniLM-L6-cos-v1"
)

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs={
        "normalize_embeddings": True
    }
)


# ----------------------------------------------------------
# 4. EMBED KNOWLEDGE BASE ONCE
# ----------------------------------------------------------

chunk_texts = [
    chunk.page_content
    for chunk in chunks
]

chunk_embeddings = np.array(
    embeddings.embed_documents(
        chunk_texts
    )
)

print(
    "Chunk embeddings created:",
    chunk_embeddings.shape
)


# ----------------------------------------------------------
# 5. LOAD LOCAL GEMMA MODEL
# ----------------------------------------------------------

llm = ChatOllama(
    model="gemma3:4b",
    temperature=0
)


# ==========================================================
# DEFAULT RAG SYSTEM PROMPT
# ==========================================================

default_system_prompt = """
You are a pre-audit assistant for a University
Accounting Office.

Answer the user's question using only the
retrieved checklist evidence.

Some retrieved chunks may be neighboring chunks
included to preserve document context.

Ignore neighboring material that is unrelated
to the user's question.

Do not invent supporting-document requirements.

Do not invent policies, procedures, amounts,
limits, or other information that is not stated
in the retrieved evidence.

Focus only on what the user explicitly asked.

If the retrieved evidence is insufficient,
clearly say that the available evidence is
insufficient.

Give a concise and practical answer.
"""


# ==========================================================
# COMPLETE RAG FUNCTION
# ==========================================================

def run_rag(
    question,
    system_prompt=default_system_prompt
):


    # ------------------------------------------------------
    # A. BM25 KEYWORD RETRIEVAL
    # ------------------------------------------------------

    query_tokens = tokenize(
        question
    )

    bm25_scores = (
        bm25_retriever.vectorizer.get_scores(
            query_tokens
        )
    )

    bm25_ranked_indices = np.argsort(
        bm25_scores
    )[::-1]


    # ------------------------------------------------------
    # B. SBERT SEMANTIC RETRIEVAL
    # ------------------------------------------------------

    query_embedding = np.array(
        embeddings.embed_query(
            question
        )
    )

    semantic_scores = (
        chunk_embeddings
        @ query_embedding
    )

    semantic_ranked_indices = np.argsort(
        semantic_scores
    )[::-1]


    # ------------------------------------------------------
    # C. CREATE RANK LOOKUPS
    # ------------------------------------------------------

    bm25_ranks = {
        int(index): rank
        for rank, index in enumerate(
            bm25_ranked_indices,
            start=1
        )
    }

    semantic_ranks = {
        int(index): rank
        for rank, index in enumerate(
            semantic_ranked_indices,
            start=1
        )
    }


    # ------------------------------------------------------
    # D. RECIPROCAL RANK FUSION
    # ------------------------------------------------------

    rrf_constant = 60

    keyword_weight = 0.50
    semantic_weight = 0.50

    hybrid_scores = np.zeros(
        len(chunks)
    )


    for rank, index in enumerate(
        bm25_ranked_indices,
        start=1
    ):

        hybrid_scores[index] += (
            keyword_weight
            / (rrf_constant + rank)
        )


    for rank, index in enumerate(
        semantic_ranked_indices,
        start=1
    ):

        hybrid_scores[index] += (
            semantic_weight
            / (rrf_constant + rank)
        )


    # ------------------------------------------------------
    # E. SELECT TOP-K HYBRID RESULTS
    # ------------------------------------------------------

    hybrid_ranked_indices = np.argsort(
        hybrid_scores
    )[::-1]

    top_k = 4

    top_indices = [
        int(index)
        for index in hybrid_ranked_indices[:top_k]
    ]

    primary_docs = [
        chunks[index]
        for index in top_indices
    ]


    # ------------------------------------------------------
    # F. NEIGHBOR EXPANSION
    # ------------------------------------------------------

    retrieved_docs, expanded_indices = (
        expand_neighbor_chunks(
            chunks,
            top_indices,
            neighbor_window=1
        )
    )


    # ------------------------------------------------------
    # G. BUILD RETRIEVED CONTEXT
    # ------------------------------------------------------

    context = "\n\n".join(
        [
            (
                f'[{doc.metadata["chunk_id"]}, '
                f'p. {doc.metadata["page_number"]}]\n'
                f'SOURCE: {doc.metadata["source_name"]}\n'
                f'TEXT: {doc.page_content}'
            )
            for doc in retrieved_docs
        ]
    )


    # ------------------------------------------------------
    # H. BUILD AUGMENTED PROMPT
    # ------------------------------------------------------

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt
            ),
            (
                "human",
                """
RETRIEVED CHECKLIST EVIDENCE:

{context}

USER QUESTION:

{question}
"""
            )
        ]
    )


    # ------------------------------------------------------
    # I. CREATE LANGCHAIN PIPELINE
    # ------------------------------------------------------

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )


    # ------------------------------------------------------
    # J. DISPLAY RETRIEVAL RESULTS
    # ------------------------------------------------------

    print("\n" + "=" * 75)

    print("USER QUESTION:")
    print(question)

    print(
        "\nPRIMARY HYBRID RETRIEVAL RESULTS:"
    )


    for hybrid_rank, index in enumerate(
        top_indices,
        start=1
    ):

        doc = chunks[index]

        print("\n" + "-" * 70)

        print(
            f'Hybrid Rank {hybrid_rank}: '
            f'[{doc.metadata["chunk_id"]}] '
            f'Page {doc.metadata["page_number"]}'
        )

        print(
            "BM25 Rank:",
            bm25_ranks[index]
        )

        print(
            "Semantic Rank:",
            semantic_ranks[index]
        )

        print(
            "RRF Score:",
            round(
                float(
                    hybrid_scores[index]
                ),
                6
            )
        )


    # ------------------------------------------------------
    # K. DISPLAY NEIGHBOR EXPANSION
    # ------------------------------------------------------

    print(
        "\nCHUNKS AFTER NEIGHBOR EXPANSION:"
    )

    print(
        [
            doc.metadata["chunk_id"]
            for doc in retrieved_docs
        ]
    )


    # ------------------------------------------------------
    # L. STREAM GEMMA RESPONSE
    # ------------------------------------------------------

    print("\n" + "=" * 75)

    print("RAG RESPONSE:")
    print("-" * 75)

    answer = ""

    for output_chunk in chain.stream(
        {
            "context": context,
            "question": question
        }
    ):

        answer += output_chunk

        print(
            output_chunk,
            end="",
            flush=True
        )

    print("\n")


    # ------------------------------------------------------
    # M. RETURN EVERYTHING NEEDED BY FUTURE EXERCISES
    # ------------------------------------------------------

    return {

        "question":
            question,

        "answer":
            answer,

        "context":
            context,

        "primary_docs":
            primary_docs,

        "retrieved_docs":
            retrieved_docs,

        "top_indices":
            top_indices,

        "expanded_indices":
            expanded_indices,

        "retrieval": {

            "bm25_scores":
                bm25_scores,

            "semantic_scores":
                semantic_scores,

            "hybrid_scores":
                hybrid_scores,

            "bm25_ranks":
                bm25_ranks,

            "semantic_ranks":
                semantic_ranks
        }
    }


# ==========================================================
# EXERCISE 5 TEST
# ==========================================================

question = """
I attended a local conference and paid the travel
expenses myself.

What supporting documents should I submit
for reimbursement?
"""


result = run_rag(
    question
)
#%% EXERCISE 6: GROUNDING AND EVIDENCE SUFFICIENCY
# Exercise 5 run_rag()
# → Retrieved Evidence
# → Gemma Evidence Evaluation
# → SUFFICIENT / PARTIAL / INSUFFICIENT
# → Compare with Expected Evaluation Label

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


if "run_rag" not in globals():

    raise RuntimeError(
        "run_rag() is not available.\n"
        "Run Exercise 5 first."
    )


if "llm" not in globals():

    raise RuntimeError(
        "The Gemma LLM is not available.\n"
        "Run Exercise 5 first."
    )


# ==========================================================
# 1. EVIDENCE-EVALUATION PROMPT
# ==========================================================

evidence_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an evidence evaluator for a University
pre-audit RAG system.

Your task is to determine whether the RETRIEVED EVIDENCE
contains enough information to answer the USER QUESTION.

IMPORTANT RULES:

1. Evaluate only what the user explicitly asked.

2. Do not create additional questions, requirements,
   conditions, or information needs that the user
   did not ask about.

3. Do not require information about meals, insurance,
   allowances, reimbursement limits, transportation,
   or other topics unless the user explicitly asks
   about them.

4. Use only the retrieved evidence.

5. Do not use your own general knowledge.

6. Do not invent University policies, documentary
   requirements, amounts, limits, or procedures.

7. Some evidence may consist of neighboring chunks
   added to preserve document context.

8. Ignore unrelated material that appears in
   neighboring chunks.

9. Treat an item as supported only when the retrieved
   evidence clearly associates it with the transaction
   or issue asked about by the user.

10. A question asking "What supporting documents are
    required?" is fully answered when the retrieved
    evidence provides the applicable supporting-document
    requirements for that transaction.

CLASSIFY THE EVIDENCE AS:

SUFFICIENT:
The retrieved evidence contains enough information
to answer every part explicitly requested by the user.

PARTIAL:
The retrieved evidence answers at least one explicit
part of the user's question, but another explicitly
requested part is missing.

INSUFFICIENT:
The retrieved evidence does not contain enough
information to answer the main question.

Do not classify evidence as PARTIAL merely because
additional related policies or details might exist.

Only missing information explicitly needed to answer
the user's actual question should affect the status.

Return only valid JSON containing exactly these fields:

status
supported_items
missing_items
reason

The status must be exactly one of:

SUFFICIENT
PARTIAL
INSUFFICIENT
"""
        ),
        (
            "human",
            """
RETRIEVED EVIDENCE:

{context}

USER QUESTION:

{question}
"""
        )
    ]
)


# ==========================================================
# 2. EVIDENCE-EVALUATION CHAIN
# ==========================================================
# Do not stream this output.
#
# JsonOutputParser needs the complete JSON response
# before the result can be evaluated.

evidence_chain = (
    evidence_prompt
    | llm
    | JsonOutputParser()
)


# ==========================================================
# 3. EVALUATION TEST CASES
# ==========================================================

test_cases = [

    {
        "question": (
            "What supporting documents are required "
            "for a training or conference held "
            "outside campus?"
        ),

        "expected_status":
            "SUFFICIENT"
    },

    {
        "question": (
            "What supporting documents are required "
            "for a conference outside campus, and is "
            "there a P10,000 maximum registration fee "
            "that can be reimbursed?"
        ),

        "expected_status":
            "PARTIAL"
    },

    {
        "question": (
            "Will the University give me a P500 daily "
            "allowance just for attending an "
            "outside-campus conference?"
        ),

        "expected_status":
            "INSUFFICIENT"
    }
]


# ==========================================================
# 4. RUN EVIDENCE-SUFFICIENCY EVALUATION
# ==========================================================

passed_tests = 0


for test_number, test_case in enumerate(
    test_cases,
    start=1
):

    question = (
        test_case["question"]
    )

    expected_status = (
        test_case["expected_status"]
    )


    # ------------------------------------------------------
    # A. RUN THE COMPLETE RAG PIPELINE
    # ------------------------------------------------------
    # run_rag() already performs:
    #
    # BM25 + SBERT
    # → RRF
    # → Top-k
    # → Neighbor Expansion
    # → Context Construction
    #
    # Exercise 6 reuses the resulting context.

    rag_result = run_rag(
        question
    )


    # ------------------------------------------------------
    # B. GET RETRIEVED EVIDENCE
    # ------------------------------------------------------

    context = (
        rag_result["context"]
    )

    retrieved_docs = (
        rag_result["retrieved_docs"]
    )


    # ------------------------------------------------------
    # C. EVALUATE EVIDENCE SUFFICIENCY
    # ------------------------------------------------------

    evaluation = evidence_chain.invoke(
        {
            "context": context,
            "question": question
        }
    )


    # ------------------------------------------------------
    # D. STANDARDIZE STATUS
    # ------------------------------------------------------

    status = str(
        evaluation.get(
            "status",
            "INSUFFICIENT"
        )
    ).strip().upper()


    valid_statuses = {
        "SUFFICIENT",
        "PARTIAL",
        "INSUFFICIENT"
    }


    if status not in valid_statuses:

        status = "INSUFFICIENT"


    # ------------------------------------------------------
    # E. COMPARE WITH EXPECTED LABEL
    # ------------------------------------------------------

    test_passed = (
        status
        == expected_status
    )


    if test_passed:

        passed_tests += 1

        evaluation_result = (
            "PASS"
        )

    else:

        evaluation_result = (
            "FAIL"
        )


    # ------------------------------------------------------
    # F. DISPLAY EVALUATION RESULTS
    # ------------------------------------------------------

    print("\n" + "=" * 75)

    print(
        f"EVIDENCE EVALUATION TEST {test_number}"
    )

    print("\nUSER QUESTION:")

    print(
        question
    )


    print(
        "\nEVIDENCE PROVIDED TO EVALUATOR:"
    )

    print(
        [
            (
                doc.metadata["chunk_id"],
                doc.metadata["page_number"]
            )
            for doc in retrieved_docs
        ]
    )


    print("\nEXPECTED STATUS:")

    print(
        expected_status
    )


    print("\nPREDICTED STATUS:")

    print(
        status
    )


    print("\nSUPPORTED ITEMS:")

    print(
        evaluation.get(
            "supported_items",
            []
        )
    )


    print("\nMISSING ITEMS:")

    print(
        evaluation.get(
            "missing_items",
            []
        )
    )


    print("\nREASON:")

    print(
        evaluation.get(
            "reason",
            ""
        )
    )


    print("\nEVALUATION RESULT:")

    print(
        evaluation_result
    )


# ==========================================================
# 5. OVERALL EVALUATION SUMMARY
# ==========================================================

accuracy = (
    passed_tests
    / len(test_cases)
)


print("\n" + "=" * 75)

print(
    "EVIDENCE SUFFICIENCY EVALUATION SUMMARY"
)


print(
    "\nTests Passed:",
    f"{passed_tests}/{len(test_cases)}"
)


print(
    "Classification Accuracy:",
    f"{accuracy:.1%}"
)
#%% EXERCISE 7: CITATIONS AND TRACEABILITY
# Exercise 5 run_rag()
# → Citation-Aware RAG Answer
# → Citation Validation
# → Traceability
import re
if "run_rag" not in globals():

    raise RuntimeError(
        "run_rag() is not available.\n"
        "Run Exercise 5 first."
    )


# ==========================================================
# HELPER FUNCTION: VALIDATE CITATIONS
# ==========================================================

def validate_citations(
    answer,
    retrieved_docs
):

    # ------------------------------------------------------
    # 1. EXTRACT CITATIONS FROM ANSWER
    # ------------------------------------------------------
    # Expected format:
    # [C023, p. 8]

    citation_matches = re.findall(
        r"\[(C\d+),\s*p\.\s*(\d+)\]",
        answer
    )

    cited_citations = set(
        citation_matches
    )


    # ------------------------------------------------------
    # 2. BUILD SET OF RETRIEVED CITATIONS
    # ------------------------------------------------------

    retrieved_citations = set(
        (
            doc.metadata["chunk_id"],
            str(doc.metadata["page_number"])
        )
        for doc in retrieved_docs
    )


    # ------------------------------------------------------
    # 3. IDENTIFY VALID AND INVALID CITATIONS
    # ------------------------------------------------------

    invalid_citations = (
        cited_citations
        - retrieved_citations
    )

    valid_citations = (
        cited_citations
        & retrieved_citations
    )


    # ------------------------------------------------------
    # 4. DETERMINE CITATION STATUS
    # ------------------------------------------------------

    if len(cited_citations) == 0:

        status = "FAILED"

        reason = (
            "The answer contains no citations."
        )


    elif len(invalid_citations) > 0:

        status = "FAILED"

        reason = (
            "One or more citations refer to a "
            "chunk or page that was not retrieved."
        )


    else:

        status = "PASSED"

        reason = (
            "All cited chunk IDs and page numbers "
            "appear in the retrieved evidence."
        )


    # ------------------------------------------------------
    # 5. RETURN RESULTS
    # ------------------------------------------------------

    return {
        "status": status,
        "reason": reason,
        "cited_citations": cited_citations,
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations
    }


# ==========================================================
# 1. CITATION-AWARE SYSTEM PROMPT
# ==========================================================

citation_system_prompt = """
You are a pre-audit assistant for a University
Accounting Office.

Answer using only the retrieved checklist evidence.

IMPORTANT RULES:

1. Answer only what the user explicitly asked.

2. Every factual documentary requirement must include
   a citation immediately after the claim.

3. Use citations exactly in this format:

   [C023, p. 8]

4. Use only chunk IDs and page numbers that appear
   in the retrieved evidence.

5. Do not invent supporting-document requirements,
   policies, limits, procedures, source IDs,
   or page numbers.

6. Some evidence consists of neighboring chunks
   included to preserve document context.

7. Ignore requirements belonging to unrelated
   transaction types even if they appear in
   neighboring chunks.

8. If the evidence is insufficient, clearly say so.

9. Keep the answer concise and practical.
"""


# ==========================================================
# 2. USER QUESTION
# ==========================================================

question = (
    "What supporting documents are required "
    "for a training or conference held outside campus?"
)


# ==========================================================
# 3. RUN CITATION-AWARE RAG
# ==========================================================
# run_rag() already performs:
#
# SQLite retrieval
# → BM25
# → SBERT
# → RRF
# → Neighbor Expansion
# → Augmented Prompt
# → Gemma
#
# We simply replace the normal Exercise 5 system prompt
# with a citation-aware system prompt.

rag_result = run_rag(
    question,
    system_prompt=citation_system_prompt
)


# ==========================================================
# 4. GET VALUES RETURNED BY run_rag()
# ==========================================================

answer = (
    rag_result["answer"]
)

context = (
    rag_result["context"]
)

retrieved_docs = (
    rag_result["retrieved_docs"]
)

primary_docs = (
    rag_result["primary_docs"]
)


# ==========================================================
# 5. VALIDATE CITATIONS
# ==========================================================

citation_check = validate_citations(
    answer,
    retrieved_docs
)


# ==========================================================
# 6. CREATE TRACEABILITY RECORDS
# ==========================================================

traceability = []


for doc in retrieved_docs:

    chunk_id = (
        doc.metadata["chunk_id"]
    )

    page_number = str(
        doc.metadata["page_number"]
    )

    citation_key = (
        chunk_id,
        page_number
    )


    traceability.append(
        {
            "chunk_id": chunk_id,

            "page":
                page_number,

            "source":
                doc.metadata[
                    "source_name"
                ],

            "cited_in_answer":
                (
                    citation_key
                    in citation_check[
                        "valid_citations"
                    ]
                )
        }
    )


# ==========================================================
# 7. DISPLAY CITATION VALIDATION
# ==========================================================

print("\n" + "=" * 75)

print("CITATION CHECK:")

print(
    citation_check["status"]
)


print("\nREASON:")

print(
    citation_check["reason"]
)


# ----------------------------------------------------------
# CITATIONS FOUND
# ----------------------------------------------------------

print("\nCITATIONS FOUND:")


if len(
    citation_check["cited_citations"]
) == 0:

    print("None")


else:

    for chunk_id, page in sorted(
        citation_check[
            "cited_citations"
        ]
    ):

        print(
            f"[{chunk_id}, p. {page}]"
        )


# ----------------------------------------------------------
# INVALID CITATIONS
# ----------------------------------------------------------

print("\nINVALID CITATIONS:")


if len(
    citation_check["invalid_citations"]
) == 0:

    print("None")


else:

    for chunk_id, page in sorted(
        citation_check[
            "invalid_citations"
        ]
    ):

        print(
            f"[{chunk_id}, p. {page}]"
        )


# ==========================================================
# 8. DISPLAY TRACEABILITY
# ==========================================================

print("\n" + "=" * 75)

print("TRACEABILITY:")


for item in traceability:

    print(
        item["chunk_id"],
        "| Page:",
        item["page"],
        "| Source:",
        item["source"],
        "| Cited:",
        item["cited_in_answer"]
    )
#%% EXERCISE 8: CONTROLLED RAG
# Exercise 5 run_rag()
# → Exercise 6 Evidence Sufficiency
# → Exercise 7 Citation Validation
# → Release / Abstain / Human Review


# ==========================================================
# REQUIREMENTS
# ==========================================================
# Run Exercises 5, 6, and 7 first.
#
# Exercise 5 provides:
#     run_rag()
#
# Exercise 6 provides:
#     evidence_chain
#
# Exercise 7 provides:
#     validate_citations()
#
# Exercise 8 adds:
#     CONTROL DECISIONS


if "run_rag" not in globals():

    raise RuntimeError(
        "run_rag() is not available.\n"
        "Run Exercise 5 first."
    )


if "evidence_chain" not in globals():

    raise RuntimeError(
        "evidence_chain is not available.\n"
        "Run Exercise 6 first."
    )


if "validate_citations" not in globals():

    raise RuntimeError(
        "validate_citations() is not available.\n"
        "Run Exercise 7 first."
    )


# ==========================================================
# 1. CONTROLLED-RAG SYSTEM PROMPT
# ==========================================================

controlled_system_prompt = """
You are a pre-audit assistant for a University
Accounting Office.

Answer using only the retrieved checklist evidence.

IMPORTANT RULES:

1. Answer only what the user explicitly asked.

2. Every factual documentary requirement must include
   a citation immediately after the claim.

3. Use citations exactly in this format:

   [C023, p. 8]

4. Use only chunk IDs and page numbers that appear
   in the retrieved evidence.

5. Do not invent supporting-document requirements,
   policies, limits, procedures, amounts,
   source IDs, or page numbers.

6. Some evidence consists of neighboring chunks
   included to preserve document context.

7. Ignore information belonging to unrelated
   transaction types.

8. If part of the question cannot be answered from
   the evidence, clearly state that the retrieved
   evidence does not establish that information.

9. Do not fill evidence gaps using general knowledge.

10. Keep the answer concise and practical.
"""


# ==========================================================
# 2. CONTROLLED RAG FUNCTION
# ==========================================================

def controlled_rag(question):


    # ------------------------------------------------------
    # A. RUN THE REUSABLE RAG PIPELINE
    # ------------------------------------------------------
    # Exercise 5 handles:
    #
    # SQLite
    # → BM25
    # → SBERT
    # → RRF
    # → Neighbor Expansion
    # → Context
    # → Citation-aware RAG draft

    rag_result = run_rag(
        question,
        system_prompt=controlled_system_prompt
    )


    answer = (
        rag_result["answer"]
    )

    context = (
        rag_result["context"]
    )

    retrieved_docs = (
        rag_result["retrieved_docs"]
    )


    # ------------------------------------------------------
    # B. EVALUATE EVIDENCE SUFFICIENCY
    # ------------------------------------------------------
    # Exercise 6 evaluator is reused here.

    evaluation = evidence_chain.invoke(
        {
            "context": context,
            "question": question
        }
    )


    # ------------------------------------------------------
    # C. STANDARDIZE STATUS
    # ------------------------------------------------------

    status = str(
        evaluation.get(
            "status",
            "INSUFFICIENT"
        )
    ).strip().upper()


    valid_statuses = {
        "SUFFICIENT",
        "PARTIAL",
        "INSUFFICIENT"
    }


    if status not in valid_statuses:

        status = "INSUFFICIENT"


    supported_items = (
        evaluation.get(
            "supported_items",
            []
        )
    )

    missing_items = (
        evaluation.get(
            "missing_items",
            []
        )
    )


    # ------------------------------------------------------
    # D. DISPLAY EVIDENCE EVALUATION
    # ------------------------------------------------------

    print("\n" + "=" * 75)

    print(
        "CONTROLLED RAG EVALUATION"
    )

    print("\nUSER QUESTION:")

    print(
        question
    )


    print("\nEVIDENCE STATUS:")

    print(
        status
    )


    print("\nSUPPORTED ITEMS:")

    print(
        supported_items
    )


    print("\nMISSING ITEMS:")

    print(
        missing_items
    )


    print("\nEVALUATION REASON:")

    print(
        evaluation.get(
            "reason",
            ""
        )
    )


    # ------------------------------------------------------
    # E. CONTROL GATE 1:
    #    INSUFFICIENT EVIDENCE
    # ------------------------------------------------------
    # The RAG function may have created an internal draft,
    # but the system does not release it.

    if status == "INSUFFICIENT":

        decision = "ABSTAIN"

        release_response = False

        human_review = True

        final_response = (
            "The retrieved checklist evidence "
            "does not contain enough information "
            "to answer this question."
        )


        print("\n" + "=" * 75)

        print(
            "CONTROL DECISION:"
        )

        print(
            decision
        )


        print(
            "\nRAG DRAFT RELEASED:"
        )

        print(
            "NO"
        )


        print(
            "\nFINAL RESPONSE:"
        )

        print(
            final_response
        )


        print(
            "\nHUMAN REVIEW REQUIRED:"
        )

        print(
            "YES"
        )


        return {
            "status":
                status,

            "citation_status":
                "NOT RUN",

            "decision":
                decision,

            "release_response":
                release_response,

            "human_review":
                human_review,

            "answer":
                final_response
        }


    # ------------------------------------------------------
    # F. CONTROL GATE 2:
    #    VALIDATE CITATIONS
    # ------------------------------------------------------

    citation_check = validate_citations(
        answer,
        retrieved_docs
    )


    citation_valid = (
        citation_check["status"]
        == "PASSED"
    )


    # ------------------------------------------------------
    # G. DISPLAY CITATION VALIDATION
    # ------------------------------------------------------

    print("\n" + "=" * 75)

    print(
        "CITATION CHECK:"
    )

    print(
        citation_check["status"]
    )


    print("\nREASON:")

    print(
        citation_check["reason"]
    )


    print("\nCITATIONS FOUND:")


    if len(
        citation_check[
            "cited_citations"
        ]
    ) == 0:

        print(
            "None"
        )


    else:

        for chunk_id, page in sorted(
            citation_check[
                "cited_citations"
            ]
        ):

            print(
                f"[{chunk_id}, p. {page}]"
            )


    print(
        "\nINVALID CITATIONS:"
    )


    if len(
        citation_check[
            "invalid_citations"
        ]
    ) == 0:

        print(
            "None"
        )


    else:

        for chunk_id, page in sorted(
            citation_check[
                "invalid_citations"
            ]
        ):

            print(
                f"[{chunk_id}, p. {page}]"
            )


    # ------------------------------------------------------
    # H. FINAL CONTROL POLICY
    # ------------------------------------------------------

    if (
        status == "SUFFICIENT"
        and citation_valid
    ):

        decision = "RELEASE"

        release_response = True

        human_review = False


    elif (
        status == "PARTIAL"
        and citation_valid
    ):

        decision = (
            "HUMAN REVIEW BEFORE RELEASE"
        )

        release_response = False

        human_review = True


    else:

        decision = (
            "DO NOT RELEASE"
        )

        release_response = False

        human_review = True


    # ------------------------------------------------------
    # I. DISPLAY FINAL DECISION
    # ------------------------------------------------------

    print("\n" + "=" * 75)

    print(
        "FINAL CONTROL DECISION:"
    )

    print(
        decision
    )


    print(
        "\nRELEASE RESPONSE:"
    )

    print(
        "YES"
        if release_response
        else "NO"
    )


    print(
        "\nHUMAN REVIEW REQUIRED:"
    )

    print(
        "YES"
        if human_review
        else "NO"
    )


    # ------------------------------------------------------
    # J. DISPLAY RESPONSE
    # ------------------------------------------------------

    if release_response:

        print(
            "\nFINAL RELEASED RESPONSE:"
        )

        print(
            answer
        )


    elif status == "PARTIAL":

        print(
            "\nDRAFT RESPONSE FOR HUMAN REVIEW:"
        )

        print(
            answer
        )


    else:

        print(
            "\nRAG DRAFT NOT RELEASED."
        )


    # ------------------------------------------------------
    # K. RETURN CONTROL RESULTS
    # ------------------------------------------------------

    return {
        "status":
            status,

        "citation_status":
            citation_check["status"],

        "decision":
            decision,

        "release_response":
            release_response,

        "human_review":
            human_review,

        "answer":
            answer
    }


# ==========================================================
# 3. CONTROLLED RAG TEST CASES
# ==========================================================

test_cases = [

    {
        "question": (
            "What supporting documents are required "
            "for a training or conference held "
            "outside campus?"
        ),

        "expected_status":
            "SUFFICIENT",

        "expected_decision":
            "RELEASE"
    },

    {
        "question": (
            "What supporting documents are required "
            "for a conference outside campus, and is "
            "there a P10,000 maximum registration fee "
            "that can be reimbursed?"
        ),

        "expected_status":
            "PARTIAL",

        "expected_decision":
            "HUMAN REVIEW BEFORE RELEASE"
    },

    {
        "question": (
            "Will the University give me a P500 daily "
            "allowance just for attending an "
            "outside-campus conference?"
        ),

        "expected_status":
            "INSUFFICIENT",

        "expected_decision":
            "ABSTAIN"
    }
]


# ==========================================================
# 4. RUN CONTROLLED-RAG TESTS
# ==========================================================

passed_status_tests = 0

passed_decision_tests = 0


for test_number, test_case in enumerate(
    test_cases,
    start=1
):

    print(
        "\n\n"
        + "#"
        * 75
    )

    print(
        f"CONTROLLED RAG TEST {test_number}"
    )

    print(
        "#"
        * 75
    )


    result = controlled_rag(
        test_case["question"]
    )


    # ------------------------------------------------------
    # EVIDENCE STATUS TEST
    # ------------------------------------------------------

    predicted_status = (
        result["status"]
    )

    expected_status = (
        test_case[
            "expected_status"
        ]
    )

    status_passed = (
        predicted_status
        == expected_status
    )


    if status_passed:

        passed_status_tests += 1


    # ------------------------------------------------------
    # CONTROL DECISION TEST
    # ------------------------------------------------------

    predicted_decision = (
        result["decision"]
    )

    expected_decision = (
        test_case[
            "expected_decision"
        ]
    )

    decision_passed = (
        predicted_decision
        == expected_decision
    )


    if decision_passed:

        passed_decision_tests += 1


    # ------------------------------------------------------
    # DISPLAY TEST RESULTS
    # ------------------------------------------------------

    print(
        "\nEXPECTED STATUS:"
    )

    print(
        expected_status
    )


    print(
        "\nPREDICTED STATUS:"
    )

    print(
        predicted_status
    )


    print(
        "\nSTATUS TEST:"
    )

    print(
        "PASS"
        if status_passed
        else "FAIL"
    )


    print(
        "\nEXPECTED CONTROL DECISION:"
    )

    print(
        expected_decision
    )


    print(
        "\nPREDICTED CONTROL DECISION:"
    )

    print(
        predicted_decision
    )


    print(
        "\nCONTROL TEST:"
    )

    print(
        "PASS"
        if decision_passed
        else "FAIL"
    )


# ==========================================================
# 5. OVERALL TEST SUMMARY
# ==========================================================

status_accuracy = (
    passed_status_tests
    / len(test_cases)
)


decision_accuracy = (
    passed_decision_tests
    / len(test_cases)
)


print(
    "\n"
    + "="
    * 75
)

print(
    "CONTROLLED RAG TEST SUMMARY"
)


print(
    "\nEvidence Status Tests Passed:",
    f"{passed_status_tests}/{len(test_cases)}"
)


print(
    "Evidence Classification Accuracy:",
    f"{status_accuracy:.1%}"
)


print(
    "\nControl Decision Tests Passed:",
    f"{passed_decision_tests}/{len(test_cases)}"
)


print(
    "Control Decision Accuracy:",
    f"{decision_accuracy:.1%}"
)