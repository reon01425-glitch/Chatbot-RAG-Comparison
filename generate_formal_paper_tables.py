import os
import pandas as pd
import numpy as np

def generate_paper_tables():
    print("=" * 70)
    print("📝 GENERATING FORMAL BENCHMARK TABLES FOR PAPER & THESIS (FSM UNDIP)")
    print("=" * 70)
    
    # Load dataset reports
    comp_file = "comparison_report.csv"
    graph_comp_file = "graph_chunking_comparison_report.csv"
    
    if not os.path.exists(comp_file):
        print(f"Error: {comp_file} not found.")
        return
        
    df = pd.read_csv(comp_file)
    
    # 1. TABLE 1: Comprehensive Benchmark of All 8 RAG Architectures
    print("\n--- TABLE 1: Master Benchmark Evaluation across 8 Architectures ---")
    df_sorted = df.sort_values(by="Ragas Faithfulness", ascending=False).reset_index(drop=True)
    print(df_sorted.to_string(index=False))
    
    # Export LaTeX Table for Paper / Skripsi
    latex_table_1 = df_sorted.to_latex(
        index=False,
        caption="Hasil Evaluasi Komparatif 8 Arsitektur RAG pada Dokumen SOP FSM UNDIP",
        label="tab:rag_benchmark_master",
        float_format="%.4f",
        column_format="lccccc"
    )
    with open("docs/table_master_benchmark.tex", "w") as f:
        f.write(latex_table_1)
    print("\nSaved LaTeX Table 1 to docs/table_master_benchmark.tex")
    
    # 2. TABLE 2: Specialized Comparison: Procedural Graph & Chunking Methods
    if os.path.exists(graph_comp_file):
        print("\n--- TABLE 2: Procedural Graph & Chunking Evaluation ---")
        df_graph = pd.read_csv(graph_comp_file)
        print(df_graph.to_string(index=False))
        latex_table_2 = df_graph.to_latex(
            index=False,
            caption="Perbandingan Kinerja Representasi Graf dan Chunking Terhadap Kueri Prosedural SOP",
            label="tab:sop_graph_chunking_comparison",
            float_format="%.4f",
            column_format="lcccccc"
        )
        with open("docs/table_graph_chunking.tex", "w") as f:
            f.write(latex_table_2)
        print("Saved LaTeX Table 2 to docs/table_graph_chunking.tex")
        
    # 3. TABLE 3: Qualitative Feature Matrix (Methodology, Chunking, Graph Type, Output)
    feature_matrix = [
        {
            "Arsitektur": "Naive RAG (Baseline)",
            "Model Retrieval": "Dense Vector Cosine",
            "Tipe Representasi": "Fixed-size (1700 chars)",
            "Struktur Graf": "None (Flat Chunks)",
            "Mitigasi Halusinasi": "Threshold Filter (0.30)",
            "Format Luaran": "Teks Naratif"
        },
        {
            "Arsitektur": "Hybrid RAG",
            "Model Retrieval": "Dense + Sparse BM25 (RRF)",
            "Tipe Representasi": "Fixed-size Tokenized",
            "Struktur Graf": "None (Inverted Index)",
            "Mitigasi Halusinasi": "Rank Fusion (k=60)",
            "Format Luaran": "Teks Naratif"
        },
        {
            "Arsitektur": "GraphRAG (Entity)",
            "Model Retrieval": "Graph Query Expansion",
            "Tipe Representasi": "Entity-Enriched Chunks",
            "Struktur Graf": "Undirected Relation Graph",
            "Mitigasi Halusinasi": "Relation Header Injection",
            "Format Luaran": "Teks + Hubungan Entitas"
        },
        {
            "Arsitektur": "Agentic RAG",
            "Model Retrieval": "ReAct Tool Execution",
            "Tipe Representasi": "On-demand Dynamic Lookup",
            "Struktur Graf": "None (Tool Invocation)",
            "Mitigasi Halusinasi": "Multi-hop Verification Loop",
            "Format Luaran": "Teks + Thought Tracing"
        },
        {
            "Arsitektur": "Corrective RAG (CRAG)",
            "Model Retrieval": "Self-Graded Adaptive",
            "Tipe Representasi": "Evaluated Confidence Chunks",
            "Struktur Graf": "None (Decision Boundary)",
            "Mitigasi Halusinasi": "Grader + Query Rewriting",
            "Format Luaran": "Teks + Confidence Grade"
        },
        {
            "Arsitektur": "Multimodal RAG",
            "Model Retrieval": "Layout-Aware Embedding",
            "Tipe Representasi": "Visual + Text Chunks",
            "Struktur Graf": "Page-Element Hierarchy",
            "Mitigasi Halusinasi": "Layout Flow Metadata",
            "Format Luaran": "Teks + Layout Metadata"
        },
        {
            "Arsitektur": "Hierarchical Tree RAG",
            "Model Retrieval": "Bottom-up Leaf Search",
            "Tipe Representasi": "Parent-Child Tree Chunker",
            "Struktur Graf": "Document Tree Hierarchy",
            "Mitigasi Halusinasi": "Parent Context Fusion",
            "Format Luaran": "Teks Terstruktur Pohon"
        },
        {
            "Arsitektur": "Workflow GraphRAG",
            "Model Retrieval": "Topological State Machine",
            "Tipe Representasi": "Procedural Step DAG",
            "Struktur Graf": "Directed Process Graph (DAG)",
            "Mitigasi Halusinasi": "Topological Step Grounding",
            "Format Luaran": "Teks + Bagan Alur BPMN/Mermaid"
        }
    ]
    df_features = pd.DataFrame(feature_matrix)
    print("\n--- TABLE 3: Qualitative Architecture Feature Comparison Matrix ---")
    print(df_features.to_string(index=False))
    
    df_features.to_csv("docs/table_qualitative_features.csv", index=False)
    latex_table_3 = df_features.to_latex(
        index=False,
        caption="Matriks Perbandingan Karakteristik Teknis 8 Arsitektur RAG untuk SOP",
        label="tab:rag_qualitative_matrix",
        column_format="lp{3cm}p{3cm}p{3cm}p{3cm}p{3cm}"
    )
    with open("docs/table_qualitative_features.tex", "w") as f:
        f.write(latex_table_3)
    print("Saved LaTeX Table 3 to docs/table_qualitative_features.tex")

if __name__ == "__main__":
    generate_paper_tables()
