import os
import re
from typing import Dict, List, Any, Optional
from langchain.schema.document import Document

class SOPHierarchicalChunker:
    """
    Hierarchical / Tree-structured chunker for SOP documents.
    Preserves Document-level metadata, Section hierarchy, Step-level procedures,
    and child attributes (Prerequisites, Estimated Duration, Forms/Links).
    """
    def __init__(self, data_path: str = "data"):
        self.data_path = data_path
        self.document_tree = {}
        self.leaf_chunks = []
        self._build_tree()

    def _build_tree(self):
        """
        Parse structured SOP text into a hierarchical tree:
        Document (Root) -> Sections (Overview, Requirements, Procedural Steps) -> Step Children
        """
        import pypdf
        
        if not os.path.exists(self.data_path):
            return
            
        for fname in sorted(os.listdir(self.data_path)):
            if not fname.endswith(".pdf"):
                continue
                
            fpath = os.path.join(self.data_path, fname)
            reader = pypdf.PdfReader(fpath)
            full_text = ""
            for p in reader.pages:
                txt = p.extract_text() or ""
                full_text += txt + "\n"
                
            lines = [l.strip() for l in full_text.split("\n") if l.strip()]
            if not lines:
                continue
                
            doc_title = lines[0]
            if len(lines) > 1 and not lines[1].startswith("Dokumen") and not lines[1][0].isdigit():
                doc_title += " " + lines[1]
                
            overview_lines = []
            step_blocks = []
            current_step = None
            
            for line in lines:
                # Check for numbered step: "1. Mahasiswa ...", "2. Dekan ...", etc.
                step_match = re.match(r"^(\d+)\.\s+(.*)", line)
                if step_match:
                    if current_step:
                        step_blocks.append(current_step)
                    step_num = int(step_match.group(1))
                    step_content = step_match.group(2)
                    current_step = {
                        "step_num": step_num,
                        "header": f"Langkah {step_num}",
                        "text": step_content,
                        "details": []
                    }
                elif current_step:
                    current_step["details"].append(line)
                else:
                    overview_lines.append(line)
                    
            if current_step:
                step_blocks.append(current_step)
                
            overview_text = " ".join(overview_lines)
            
            doc_node = {
                "doc_name": fname,
                "title": doc_title,
                "overview": overview_text,
                "steps": step_blocks
            }
            self.document_tree[fname] = doc_node
            
            # Create Leaf Chunks enriched with Parent Context (Tree Context Fusion)
            # 1. Overview chunk
            overview_chunk = Document(
                page_content=f"DOKUMEN: {doc_title}\nBAGIAN: Ketentuan Umum & Ringkasan SOP\nDESKRIPSI: {overview_text}",
                metadata={
                    "source": fname,
                    "level": "document_overview",
                    "doc_title": doc_title,
                    "id": f"{fname}:overview"
                }
            )
            self.leaf_chunks.append(overview_chunk)
            
            # 2. Step chunks with parent context injection
            for s in step_blocks:
                detail_str = "\n".join(s["details"]) if s["details"] else ""
                full_step_text = f"DOKUMEN INDUK: {doc_title}\nBAGIAN: {s['header']}\nISI PROSEDUR: {s['text']}"
                if detail_str:
                    full_step_text += f"\nDETAIL & ATRIBUT:\n{detail_str}"
                    
                step_chunk = Document(
                    page_content=full_step_text,
                    metadata={
                        "source": fname,
                        "level": "procedural_step",
                        "step_num": s["step_num"],
                        "doc_title": doc_title,
                        "id": f"{fname}:step_{s['step_num']}"
                    }
                )
                self.leaf_chunks.append(step_chunk)

    def get_hierarchical_context_for_doc(self, doc_name: str) -> str:
        """
        Assemble the complete hierarchical tree context for a given document.
        """
        node = self.document_tree.get(doc_name)
        if not node:
            return ""
            
        res = [f"=== STRUKTUR POHON DOKUMEN SOP: {node['title']} ==="]
        res.append(f"Ringkasan: {node['overview']}\n")
        res.append("Rincian Sub-Langkah Pohon (Child Steps):")
        for s in node["steps"]:
            details = " | ".join(s["details"]) if s["details"] else ""
            res.append(f"├── [Langkah {s['step_num']}] {s['text']}")
            if details:
                res.append(f"│   └── Atribut: {details}")
        return "\n".join(res)
