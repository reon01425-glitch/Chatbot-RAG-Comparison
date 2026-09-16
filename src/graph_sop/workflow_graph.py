import os
import re
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple

class SOPWorkflowGraph:
    """
    Process / Workflow Directed Graph (DiGraph) representing procedural steps,
    actors (swimlanes), requirements, and artifacts in SOP FSM UNDIP.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.sop_metadata = {}
        self._build_workflow_knowledge_base()

    def _build_workflow_knowledge_base(self):
        """
        Populate the workflow directed graph with procedural step sequences,
        actor assignments, durations, and output artifacts from SOP documents.
        """
        sop_definitions = [
            {
                "id": "SOP_CUTI_AKADEMIK",
                "title": "Permohonan Izin Cuti Akademik",
                "aliases": ["cuti akademik", "izin cuti", "berhenti sementara kuliah", "cuti kuliah"],
                "max_duration": "3 hari kerja",
                "steps": [
                    {
                        "step_num": 1,
                        "actor": "Mahasiswa",
                        "action": "Mengunduh, mengisi, dan menandatangani Form Cuti Akademik serta mengisi online via SIAP",
                        "inputs": ["Form Cuti Akademik", "Transkrip Akademik", "Bukti Bayar SPP/UKT Terakhir", "Fotokopi KTM"],
                        "output": "Form Cuti Terisi & Berkas Persyaratan",
                        "link": "https://drive.google.com/file/d/1miPHNvRj6XO6LEhFCiC_QXcMtynCPWR2/edit"
                    },
                    {
                        "step_num": 2,
                        "actor": "Ketua Program Studi",
                        "action": "Meminta persetujuan dan tanda tangan persetujuan cuti akademik dari Kaprodi",
                        "inputs": ["Form Cuti Terisi & Berkas"],
                        "output": "Form Cuti Ditandatangani Kaprodi"
                    },
                    {
                        "step_num": 3,
                        "actor": "Dekan",
                        "action": "Menyerahkan Form Cuti yang telah ditandatangani Kaprodi ke Dekan, lalu didisposisikan ke Subbag Akademik",
                        "inputs": ["Form Cuti Rekomendasi Kaprodi"],
                        "output": "Disposisi Dekan ke Subbag Akademik"
                    },
                    {
                        "step_num": 4,
                        "actor": "Subbag Akademik dan Kemahasiswaan",
                        "action": "Menerima, memeriksa, dan meneliti kelengkapan persyaratan berkas cuti. Memberi paraf jika lengkap",
                        "inputs": ["Form Cuti & Persyaratan"],
                        "output": "Paraf Subbag Akademik"
                    },
                    {
                        "step_num": 5,
                        "actor": "Dekan",
                        "action": "Dekan menandatangani Surat Izin Dekan perihal cuti akademik",
                        "inputs": ["Berkas Lengkap Berparaf"],
                        "output": "Surat Izin Dekan (SK Cuti)"
                    },
                    {
                        "step_num": 6,
                        "actor": "Subbag Sumber Daya",
                        "action": "Memberi nomor surat dan mengirimkan ke Subbag Akademik dengan tembusan Dosen Wali & Kaprodi",
                        "inputs": ["Surat Izin Dekan"],
                        "output": "Surat Izin Dekan Bernomor Resmi"
                    },
                    {
                        "step_num": 7,
                        "actor": "Subbag Akademik dan Kemahasiswaan",
                        "action": "Mengupdate status mahasiswa di Sistem Informasi Akademik, mengarsipkan surat, dan meneruskan ke loket",
                        "inputs": ["Surat Izin Dekan Bernomor"],
                        "output": "Status Cuti Terupdate di Sistem Akademik"
                    },
                    {
                        "step_num": 8,
                        "actor": "Mahasiswa",
                        "action": "Mahasiswa mengambil Surat Izin Dekan di loket akademik",
                        "inputs": ["Bukti Identitas Diri"],
                        "output": "Surat Izin Dekan Resmi Diterima Mahasiswa"
                    }
                ]
            },
            {
                "id": "SOP_LEGALISIR",
                "title": "Legalisir Ijazah Dan Transkrip",
                "aliases": ["legalisir", "legalisir ijazah", "legalisir transkrip", "akreditasi program studi"],
                "max_duration": "3 hari kerja",
                "steps": [
                    {
                        "step_num": 1,
                        "actor": "Alumni",
                        "action": "Menyerahkan fotokopi ijazah/transkrip/sertifikat akreditasi beserta dokumen asli ke loket Subbag Akademik",
                        "inputs": ["Fotokopi Ijazah / Transkrip", "Dokumen Asli"],
                        "output": "Berkas Pengajuan Legalisir",
                        "duration": "±5 menit"
                    },
                    {
                        "step_num": 2,
                        "actor": "Petugas Subbag Akademik",
                        "action": "Memeriksa keabsahan dan kesesuaian fotokopi dengan dokumen asli, serta memberikan cap legalisir",
                        "inputs": ["Berkas Pengajuan & Asli"],
                        "output": "Persetujuan Legalisir & Cap Dokumen",
                        "duration": "±10 menit"
                    },
                    {
                        "step_num": 3,
                        "actor": "Supervisor Akademik",
                        "action": "Memeriksa dokumen dan memberi paraf pengesahan di samping kanan nama Dekan",
                        "inputs": ["Dokumen Ber-cap"],
                        "output": "Paraf Supervisor Akademik",
                        "duration": "±15 menit"
                    },
                    {
                        "step_num": 4,
                        "actor": "Dekan / Wakil Dekan I",
                        "action": "Menandatangani lembar pengesahan kebenaran dan kesesuaian dokumen dengan aslinya",
                        "inputs": ["Dokumen Berparaf"],
                        "output": "Tanda Tangan Pengesahan Dekan/WD I",
                        "duration": "±1 hari"
                    },
                    {
                        "step_num": 5,
                        "actor": "Petugas Subbag Akademik",
                        "action": "Mengambil dokumen bertandatangan Dekan/WD I dan membubuhkan stempel resmi Fakultas Sains dan Matematika",
                        "inputs": ["Dokumen Bertanda Tangan"],
                        "output": "Stempel Fakultas Sains dan Matematika (FSM)",
                        "duration": "±5 menit"
                    },
                    {
                        "step_num": 6,
                        "actor": "Alumni",
                        "action": "Mengambil dokumen legalisir di Subbag Akademik dengan memperlihatkan dokumen asli dan menandatangani buku ekspedisi",
                        "inputs": ["Dokumen Asli", "Tanda Tangan Buku Ambil"],
                        "output": "Legalisir Ijazah / Transkrip Selesai",
                        "duration": "±15 menit"
                    }
                ]
            },
            {
                "id": "SOP_PENGISIAN_IRS",
                "title": "Pengisian Isian Rencana Studi (IRS)",
                "aliases": ["pengisian irs", "isi irs", "rencana studi", "konsultasi dosen wali", "her-registrasi"],
                "max_duration": "3 hari kerja",
                "steps": [
                    {
                        "step_num": 1,
                        "actor": "Mahasiswa",
                        "action": "Membayar biaya pendidikan (SPP/UKT) melalui bank mitra resmi Undip (BNI, Mandiri, BTN, BRI)",
                        "inputs": ["Nomor Tagihan Pembayaran"],
                        "output": "Bukti Pembayaran SPP/UKT",
                        "duration": "±10 menit"
                    },
                    {
                        "step_num": 2,
                        "actor": "Mahasiswa",
                        "action": "Melakukan her-registrasi online melalui portal SIAP Undip",
                        "inputs": ["Bukti Bayar Bank"],
                        "output": "Status Mahasiswa Aktif di SIAP",
                        "duration": "±10 menit"
                    },
                    {
                        "step_num": 3,
                        "actor": "Mahasiswa",
                        "action": "Melakukan pemilihan mata kuliah, perbaikan rencana studi, atau mencetak draf IRS sementara melalui SIAP",
                        "inputs": ["Jadwal Kuliah & Kuota Kelas"],
                        "output": "Draf IRS Sementara",
                        "duration": "±10 menit"
                    },
                    {
                        "step_num": 4,
                        "actor": "Pembimbing Akademik (Dosen Wali)",
                        "action": "Konsultasi pengambilan mata kuliah dan jumlah SKS sesuai ketentuan IPK semester sebelumnya",
                        "inputs": ["Draf IRS Sementara & KHS"],
                        "output": "Catatan Arahan Dosen Wali",
                        "duration": "±10 menit"
                    },
                    {
                        "step_num": 5,
                        "actor": "Pembimbing Akademik (Dosen Wali)",
                        "action": "Melakukan persetujuan (approval) online IRS mahasiswa pada sistem SIAP",
                        "inputs": ["Draf IRS yang Disepakati"],
                        "output": "Persetujuan Online (Approved SIAP)",
                        "duration": "±10 menit"
                    },
                    {
                        "step_num": 6,
                        "actor": "Mahasiswa",
                        "action": "Melakukan pengecekan akhir status persetujuan IRS dan mencetak dokumen IRS resmi bila diperlukan",
                        "inputs": ["Akun SIAP"],
                        "output": "File IRS Resmi Approved",
                        "duration": "±10 menit"
                    }
                ]
            },
            {
                "id": "SOP_KETERLAMBATAN_UKT",
                "title": "Permohonan Izin Keterlambatan Pembayaran UKT",
                "aliases": ["keterlambatan ukt", "izin terlambat ukt", "dispensasi ukt", "tenggang ukt"],
                "max_duration": "3 hari kerja",
                "steps": [
                    {
                        "step_num": 1,
                        "actor": "Mahasiswa",
                        "action": "Mengunduh, mengisi, dan menandatangani Form Permohonan Keterlambatan Pembayaran UKT serta melampirkan berkas alasan",
                        "inputs": ["Form Permohonan", "Surat Pernyataan / Bukti Kendala"],
                        "output": "Form Permohonan Terisi",
                        "link": "https://drive.google.com/file/d/1o7hogihjZsFTB_V27JU1OfNGBuI2Sr55/"
                    },
                    {
                        "step_num": 2,
                        "actor": "Dosen Wali & Kaprodi",
                        "action": "Meminta tanda tangan persetujuan dari Dosen Wali dan Ketua Program Studi",
                        "inputs": ["Form Permohonan Terisi"],
                        "output": "Persetujuan Dosen Wali & Kaprodi"
                    },
                    {
                        "step_num": 3,
                        "actor": "Supervisor Sumber Daya",
                        "action": "Menyerahkan formulir kepada Supervisor Sumber Daya untuk diperiksa kelengkapannya dan diberi paraf",
                        "inputs": ["Formulir Lengkap"],
                        "output": "Paraf Supervisor Sumber Daya"
                    },
                    {
                        "step_num": 4,
                        "actor": "Wakil Dekan Sumber Daya (WD II)",
                        "action": "Wakil Dekan Sumber Daya memproses dan menandatangani Surat Permohonan Izin Keterlambatan Pembayaran UKT",
                        "inputs": ["Berkas Berparaf"],
                        "output": "Surat Permohonan Bertanda Tangan WD II"
                    },
                    {
                        "step_num": 5,
                        "actor": "Subbag Sumber Daya",
                        "action": "Memproses penomoran surat permohonan keterlambatan UKT",
                        "inputs": ["Surat WD II"],
                        "output": "Surat Permohonan Bernomor Resmi"
                    },
                    {
                        "step_num": 6,
                        "actor": "Mahasiswa",
                        "action": "Membawa Surat Permohonan ke Wakil Rektor II Universitas Diponegoro melalui Manajer Akademik",
                        "inputs": ["Surat Permohonan Resmi"],
                        "output": "Penyerahan ke WR II Undip"
                    },
                    {
                        "step_num": 7,
                        "actor": "Wakil Rektor II",
                        "action": "Mendisposisikan persetujuan dispensasi ke Direktorat Keuangan Undip untuk membuka tagihan",
                        "inputs": ["Surat Permohonan"],
                        "output": "Disposisi WR II ke Ditkeu"
                    },
                    {
                        "step_num": 8,
                        "actor": "Bendahara Penerimaan UKT",
                        "action": "Membuka sistem pembayaran tagihan UKT mahasiswa yang bersangkutan",
                        "inputs": ["Disposisi Ditkeu"],
                        "output": "Tagihan Pembayaran Terbuka di Sistem Bank"
                    },
                    {
                        "step_num": 9,
                        "actor": "Mahasiswa",
                        "action": "Membayar biaya UKT ke bank mitra sebelum batas akhir perpanjangan waktu dispensasi",
                        "inputs": ["Tagihan Bank Terbuka"],
                        "output": "Bukti Lunas UKT & Status Aktif"
                    }
                ]
            },
            {
                "id": "SOP_AKTIF_SETELAH_CUTI",
                "title": "Permohonan Izin Aktif Kuliah Setelah Cuti",
                "aliases": ["aktif setelah cuti", "aktif kembali", "selesai cuti", "lapor aktif"],
                "max_duration": "1 hari kerja",
                "steps": [
                    {
                        "step_num": 1,
                        "actor": "Mahasiswa",
                        "action": "Membawa Surat Izin Cuti Akademik semester sebelumnya dan fotokopi KTM, melapor ke Kaprodi dan Subbag Akademik",
                        "inputs": ["Surat Izin Cuti Akademik", "Fotokopi KTM"],
                        "output": "Laporan Status Cuti Berakhir",
                        "duration": "±5 menit"
                    },
                    {
                        "step_num": 2,
                        "actor": "Mahasiswa",
                        "action": "Melakukan registrasi online aktivasi pada Single Sign On (SSO) masing-masing mahasiswa",
                        "inputs": ["Akun SSO Mahasiswa"],
                        "output": "Aktivasi Akun di SSO",
                        "duration": "±5 menit"
                    },
                    {
                        "step_num": 3,
                        "actor": "Subbag Akademik",
                        "action": "Memperbarui status mahasiswa sehingga terdaftar kembali sebagai Peserta Kuliah / Mahasiswa Aktif FSM",
                        "inputs": ["Konfirmasi SSO & Laporan"],
                        "output": "Status Resmi Mahasiswa Aktif Kembali",
                        "duration": "±5 menit"
                    }
                ]
            },
            {
                "id": "SOP_REKOMENDASI_BEASISWA",
                "title": "Surat Pengajuan Rekomendasi Beasiswa",
                "aliases": ["rekomendasi beasiswa", "surat beasiswa", "syarat beasiswa", "pengajuan beasiswa"],
                "max_duration": "3 hari 45 menit",
                "steps": [
                    {
                        "step_num": 1,
                        "actor": "Mahasiswa",
                        "action": "Mengunduh dan mengisi formulir permohonan rekomendasi beasiswa serta melampirkan KHS berlegalisir ke BAK Fakultas",
                        "inputs": ["Form Rekomendasi Beasiswa", "KHS Dilegalisir"],
                        "output": "Berkas Pengajuan Rekomendasi",
                        "duration": "±30 menit",
                        "link": "https://drive.google.com/file/d/18f_nbPXHbElgRNoplDFQVNYGGDkFxBTT/view?usp=sharing"
                    },
                    {
                        "step_num": 2,
                        "actor": "BAK Fakultas",
                        "action": "Meneliti dan memverifikasi kesesuaian dokumen permohonan beasiswa dan keabsahan KHS",
                        "inputs": ["Berkas Pengajuan"],
                        "output": "Hasil Verifikasi Berkas",
                        "duration": "±1 hari"
                    },
                    {
                        "step_num": 3,
                        "actor": "BAK Fakultas",
                        "action": "Memberikan paraf dan nomor surat resmi pada Surat Rekomendasi Pengajuan Beasiswa",
                        "inputs": ["Surat Terverifikasi"],
                        "output": "Surat Bernomor & Berparaf",
                        "duration": "±10 menit"
                    },
                    {
                        "step_num": 4,
                        "actor": "Wakil Dekan I (Akademik & Kemahasiswaan)",
                        "action": "Membubuhkan tanda tangan persetujuan resmi pada Surat Rekomendasi Beasiswa",
                        "inputs": ["Surat Berparaf"],
                        "output": "Tanda Tangan Surat Rekomendasi",
                        "duration": "±2 hari"
                    },
                    {
                        "step_num": 5,
                        "actor": "Mahasiswa",
                        "action": "Mengambil Surat Rekomendasi di loket BAK Fakultas dan menandatangani bukti pengambilan dokumen",
                        "inputs": ["KTM", "Tanda Tangan Buku Ambil"],
                        "output": "Surat Rekomendasi Beasiswa Sah",
                        "duration": "±5 menit"
                    }
                ]
            },
            {
                "id": "SOP_PROPOSAL_ORMAWA",
                "title": "Pengajuan Proposal Kegiatan Organisasi Mahasiswa",
                "aliases": ["proposal ormawa", "proposal kegiatan", "organisasi mahasiswa", "izin kegiatan"],
                "max_duration": "3 hari kerja",
                "steps": [
                    {
                        "step_num": 1,
                        "actor": "Mahasiswa (Pengurus Ormawa)",
                        "action": "Menyerahkan berkas proposal kegiatan ke petugas kemahasiswaan dan Supervisor Akademik untuk registrasi & alokasi ruang FSM",
                        "inputs": ["Proposal Kegiatan Organisasi"],
                        "output": "Registrasi Proposal Masuk"
                    },
                    {
                        "step_num": 2,
                        "actor": "Supervisor Akademik dan Kemahasiswaan",
                        "action": "Meneliti, memeriksa kesesuaian proposal dengan regulasi fakultas, dan memberikan paraf pengesahan",
                        "inputs": ["Proposal Registrasi"],
                        "output": "Paraf Supervisor Akademik"
                    },
                    {
                        "step_num": 3,
                        "actor": "Wakil Dekan I",
                        "action": "Mengevaluasi kelayakan substansi, anggaran, dan relevansi akademis proposal kegiatan",
                        "inputs": ["Proposal Berparaf"],
                        "output": "Persetujuan Substansi WD I"
                    },
                    {
                        "step_num": 4,
                        "actor": "Wakil Dekan I",
                        "action": "Pengesahan resmi dan tanda tangan persetujuan kegiatan oleh Wakil Dekan I",
                        "inputs": ["Lembar Pengesahan"],
                        "output": "Proposal Disahkan"
                    },
                    {
                        "step_num": 5,
                        "actor": "Mahasiswa (Pengurus Ormawa)",
                        "action": "Mengambil proposal yang telah disahkan untuk pelaksanaan kegiatan ormawa",
                        "inputs": ["Tanda Terima Pengambilan"],
                        "output": "Proposal Kegiatan Resmi Disetujui"
                    }
                ]
            }
        ]

        # Insert nodes and directed edges (Process DAG)
        for sop in sop_definitions:
            sop_id = sop["id"]
            self.sop_metadata[sop_id] = sop
            
            # Root SOP node
            self.graph.add_node(
                sop_id,
                type="SOP_ROOT",
                title=sop["title"],
                aliases=sop["aliases"],
                max_duration=sop["max_duration"]
            )
            
            prev_step_node = None
            for s in sop["steps"]:
                step_node_id = f"{sop_id}_STEP_{s['step_num']}"
                self.graph.add_node(
                    step_node_id,
                    type="SOP_STEP",
                    sop_id=sop_id,
                    sop_title=sop["title"],
                    step_num=s["step_num"],
                    actor=s["actor"],
                    action=s["action"],
                    inputs=s.get("inputs", []),
                    output=s.get("output", ""),
                    duration=s.get("duration", "-"),
                    link=s.get("link", "")
                )
                
                # Hierarchy Edge from Root to Step
                self.graph.add_edge(sop_id, step_node_id, relation="HAS_STEP")
                
                # Sequential Workflow Edge (Step i -> Step i+1)
                if prev_step_node:
                    self.graph.add_edge(prev_step_node, step_node_id, relation="NEXT_STEP")
                prev_step_node = step_node_id
                
                # Actor Edge
                actor_node_id = f"ACTOR_{s['actor'].upper().replace(' ', '_')}"
                if not self.graph.has_node(actor_node_id):
                    self.graph.add_node(actor_node_id, type="ACTOR", name=s["actor"])
                self.graph.add_edge(actor_node_id, step_node_id, relation="PERFORMS")

    def match_sop(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Identify the most matching SOP from user query via alias, keyword specificity,
        and semantic overlap.
        """
        q_lower = query.lower()
        best_sop = None
        best_score = 0
        
        # Priority rules for compound phrases
        if "aktif" in q_lower and ("cuti" in q_lower or "kembali" in q_lower):
            return self.sop_metadata.get("SOP_AKTIF_SETELAH_CUTI")
            
        for sop_id, meta in self.sop_metadata.items():
            score = 0
            # Direct title match
            if meta["title"].lower() in q_lower:
                score += 15
            # Alias match
            for alias in meta["aliases"]:
                if alias in q_lower:
                    score += 8
                else:
                    # Token overlap
                    tokens = [t for t in alias.split() if len(t) > 2]
                    matched_toks = sum(1 for t in tokens if t in q_lower)
                    score += matched_toks * 2.0
                    
            # Check step actions overlap
            for s in meta["steps"]:
                if any(k in q_lower for k in ["syarat", "cara", "alur", "prosedur", "bagaimana", "dokumen"]):
                    # If query asks about a specific action
                    if any(w in s["action"].lower() for w in q_lower.split() if len(w) > 4):
                        score += 1.0
                        
            if score > best_score:
                best_score = score
                best_sop = meta
                
        if best_score >= 3:
            return best_sop
        return None

    def get_sequential_steps(self, sop_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve ordered workflow steps using graph traversal.
        """
        sop = self.sop_metadata.get(sop_id)
        if not sop:
            return []
        return sop["steps"]

    def generate_mermaid_flowchart(self, sop_id: str, mode: str = "standard") -> str:
        """
        Generate an elegant, professional Mermaid diagram string representing the SOP workflow/BPMN.
        Modes:
          - 'standard': Vertical sequential process diagram with actor badges & artifact arrows.
          - 'swimlane': BPMN swimlane layout grouping steps by organizational actor units.
          - 'detailed': Complete flowchart including duration, input prerequisites, and output files.
        """
        sop = self.sop_metadata.get(sop_id)
        if not sop:
            return ""
            
        steps = sop["steps"]
        
        if mode == "swimlane":
            return self._generate_swimlane_mermaid(sop)
        elif mode == "detailed":
            return self._generate_detailed_mermaid(sop)
        else:
            return self._generate_standard_mermaid(sop)

    def _generate_standard_mermaid(self, sop: Dict[str, Any]) -> str:
        steps = sop["steps"]
        mermaid = ["flowchart TD"]
        mermaid.append(f'    Start(["🚀 Mulai: {sop["title"]}"])')
        
        step_nodes = []
        for s in steps:
            s_id = f"S{s['step_num']}"
            step_nodes.append(s_id)
            
            action_snippet = s['action']
            if len(action_snippet) > 60:
                action_snippet = action_snippet[:57] + "..."
            action_snippet = action_snippet.replace('"', "'")
            actor_name = s['actor'].replace('"', "'")
            
            node_label = f"<b>Langkah {s['step_num']}: {actor_name}</b><br>{action_snippet}"
            if s.get("duration") and s.get("duration") != "-":
                node_label += f"<br><i>⏱️ {s['duration']}</i>"
                
            mermaid.append(f'    {s_id}["{node_label}"]')

        mermaid.append(f'    Finish(["✅ Selesai ({sop["max_duration"]})"])')
        
        # Connect sequential edges
        mermaid.append(f"    Start --> {step_nodes[0]}")
        for i in range(len(step_nodes) - 1):
            curr = step_nodes[i]
            nxt = step_nodes[i+1]
            out_label = steps[i].get('output', '')
            if out_label and len(out_label) < 28:
                clean_lbl = out_label.replace('"', "'")
                mermaid.append(f'    {curr} -->|"{clean_lbl}"| {nxt}')
            else:
                mermaid.append(f"    {curr} --> {nxt}")
                
        mermaid.append(f"    {step_nodes[-1]} --> Finish")
        
        # Styling classes (Clean, modern university theme)
        mermaid.append("    classDef startFinish fill:#0F2C59,stroke:#3B82F6,stroke-width:2px,color:#FFFFFF,font-weight:bold;")
        mermaid.append("    classDef process fill:#F8FAFC,stroke:#3B82F6,stroke-width:1.5px,color:#0F172A;")
        mermaid.append("    class Start,Finish startFinish;")
        for s_id in step_nodes:
            mermaid.append(f"    class {s_id} process;")
            
        return "\n".join(mermaid)

    def _generate_swimlane_mermaid(self, sop: Dict[str, Any]) -> str:
        """
        Generates a true BPMN-style Swimlane diagram grouped by organizational roles/actors.
        """
        steps = sop["steps"]
        actors = []
        for s in steps:
            if s["actor"] not in actors:
                actors.append(s["actor"])
                
        mermaid = ["flowchart TB"]
        mermaid.append(f'    Start(["🚀 Mulai: {sop["title"]}"])')
        
        # Create subgraphs for each actor swimlane
        for idx, act in enumerate(actors):
            sub_id = f"Lane_{idx}"
            clean_act = act.replace('"', "'")
            mermaid.append(f'    subgraph {sub_id} ["👤 Swimlane: {clean_act}"]')
            for s in steps:
                if s["actor"] == act:
                    s_id = f"Step_{s['step_num']}"
                    snippet = s['action'][:50] + "..." if len(s['action']) > 53 else s['action']
                    snippet = snippet.replace('"', "'")
                    mermaid.append(f'        {s_id}["<b>Langkah {s["step_num"]}</b><br>{snippet}"]')
            mermaid.append('    end')
            
        mermaid.append(f'    Finish(["✅ Selesai ({sop["max_duration"]})"])')
        
        # Connect sequential edges across swimlanes
        mermaid.append(f"    Start --> Step_1")
        for i in range(1, len(steps)):
            curr = f"Step_{i}"
            nxt = f"Step_{i+1}"
            out_label = steps[i-1].get('output', '')
            if out_label and len(out_label) < 25:
                clean_lbl = out_label.replace('"', "'")
                mermaid.append(f'    {curr} -->|"{clean_lbl}"| {nxt}')
            else:
                mermaid.append(f"    {curr} --> {nxt}")
                
        mermaid.append(f"    Step_{len(steps)} --> Finish")
        
        # Styling
        mermaid.append("    classDef startFinish fill:#0F2C59,stroke:#3B82F6,stroke-width:2px,color:#FFFFFF,font-weight:bold;")
        mermaid.append("    classDef laneStep fill:#FFFFFF,stroke:#0284C7,stroke-width:1.5px,color:#0F172A;")
        mermaid.append("    class Start,Finish startFinish;")
        for s in steps:
            mermaid.append(f"    class Step_{s['step_num']} laneStep;")
            
        return "\n".join(mermaid)

    def _generate_detailed_mermaid(self, sop: Dict[str, Any]) -> str:
        """
        Generates detailed BPMN-like diagram showing input requisites, duration, and artifact deliverables.
        """
        steps = sop["steps"]
        mermaid = ["flowchart TD"]
        mermaid.append(f'    Start(["🚀 Alur Lengkap: {sop["title"]}<br><i>Total Batas Waktu: {sop["max_duration"]}</i>"])')
        
        step_nodes = []
        for s in steps:
            s_id = f"DS_{s['step_num']}"
            step_nodes.append(s_id)
            
            actor = s['actor'].replace('"', "'")
            action = s['action'].replace('"', "'")
            if len(action) > 65:
                action = action[:62] + "..."
                
            details = [f"<b>Langkah {s['step_num']}: {actor}</b>", action]
            if s.get("inputs"):
                in_str = ", ".join(s["inputs"])
                if len(in_str) > 35: in_str = in_str[:32] + "..."
                clean_in = in_str.replace('"', "'")
                details.append(f"📥 Syarat: {clean_in}")
            if s.get("output"):
                out_str = s["output"]
                if len(out_str) > 35: out_str = out_str[:32] + "..."
                clean_out = out_str.replace('"', "'")
                details.append(f"📤 Output: {clean_out}")
            if s.get("duration") and s.get("duration") != "-":
                details.append(f"⏱️ Waktu: {s['duration']}")
                
            label = "<br>".join(details)
            mermaid.append(f'    {s_id}["{label}"]')
            
        mermaid.append(f'    Finish(["✅ Proses Birokrasi Selesai"])')
        
        # Connect sequential edges
        mermaid.append(f"    Start --> {step_nodes[0]}")
        for i in range(len(step_nodes) - 1):
            mermaid.append(f"    {step_nodes[i]} --> {step_nodes[i+1]}")
        mermaid.append(f"    {step_nodes[-1]} --> Finish")
        
        # Styling
        mermaid.append("    classDef startFinish fill:#0F2C59,stroke:#3B82F6,stroke-width:2px,color:#FFFFFF,font-weight:bold;")
        mermaid.append("    classDef detailedStep fill:#F8FAFC,stroke:#4F46E5,stroke-width:1.5px,color:#1E293B;")
        mermaid.append("    class Start,Finish startFinish;")
        for s_id in step_nodes:
            mermaid.append(f"    class {s_id} detailedStep;")
            
        return "\n".join(mermaid)

    def get_workflow_context(self, sop_id: str) -> str:
        """
        Produce high-fidelity structured sequential context for LLM prompt.
        """
        sop = self.sop_metadata.get(sop_id)
        if not sop:
            return ""
            
        lines = []
        lines.append(f"=== STRUKTUR WORKFLOW ALUR RESMI: {sop['title'].upper()} ===")
        lines.append(f"Maksimal Waktu Pemrosesan: {sop['max_duration']}")
        lines.append("\nTahapan Prosedural Berurutan:")
        
        for s in sop["steps"]:
            lines.append(f"{s['step_num']}. Pelaksana / Aktor: {s['actor']}")
            lines.append(f"   Aksi: {s['action']}")
            if s.get("inputs"):
                lines.append(f"   Dokumen Prasyarat: {', '.join(s['inputs'])}")
            if s.get("output"):
                lines.append(f"   Luaran / Output: {s['output']}")
            if s.get("duration") and s.get("duration") != "-":
                lines.append(f"   Estimasi Waktu: {s['duration']}")
            if s.get("link"):
                lines.append(f"   Tautan Dokumen: {s['link']}")
            lines.append("")
            
        return "\n".join(lines)
