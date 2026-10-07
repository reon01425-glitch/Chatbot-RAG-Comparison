import os
import re
import json
import hashlib
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
        # Every step mirrors one numbered step in the SOP PDF (data/<source_pdf>):
        #   actor    = the party that PERFORMS the step (grammatical subject of the PDF sentence);
        #              officials who are only addressed (e.g. "meminta tanda tangan Kaprodi") stay in `action`.
        #   inputs   = only the PDF's "Dokumen yang dibutuhkan" list for that step (else empty).
        #   output   = only the PDF's "Output:" line for that step (else empty).
        #   duration = only the PDF's "Waktu:" line for that step (else absent).
        #   evidence = verbatim quote from the PDF (whitespace-normalised), checked by tests.
        # See docs/GRAPH_PDF_RECONCILIATION.md; the pre-reconciliation graph is in docs/workflow_graph_v1.json.
        sop_definitions = [
            {
                "id": "SOP_CUTI_AKADEMIK",
                "title": "Permohonan Izin Cuti Akademik",
                "source_pdf": "SOP_Izin_Cuti_Akademik.pdf",
                "aliases": ["cuti akademik", "izin cuti", "berhenti sementara kuliah", "cuti kuliah"],
                "max_duration": "3 hari kerja",
                "max_duration_evidence": "total maksimal waktu pemrosesan maksimal 3 (tiga) hari kerja",
                "steps": [
                    {
                        "step_num": 1,
                        "actor": "Mahasiswa",
                        "action": "Mengunduh, mengisi, dan menandatangani Form Cuti Akademik dengan melampirkan persyaratan, serta wajib mengisi secara online melalui SIAP",
                        "inputs": ["Form Cuti Akademik", "Transkrip Akademik", "Bukti bayar SPP/UKT terakhir", "Fotokopi KTM", "Dokumen pendukung lain"],
                        "output": "",
                        "link": "https://drive.google.com/file/d/1miPHNvRj6XO6LEhFCiC_QXcMtynCPWR2/edit",
                        "evidence": "Mahasiswa mengunduh, mengisi, dan menandatangani Form Cuti Akademik dengan melampirkan persyaratan."
                    },
                    {
                        "step_num": 2,
                        "actor": "Mahasiswa",
                        "action": "Meminta persetujuan dan tanda tangan Ketua Program Studi",
                        "inputs": [],
                        "output": "",
                        "evidence": "Mahasiswa meminta persetujuan dan tanda tangan Ketua Program Studi."
                    },
                    {
                        "step_num": 3,
                        "actor": "Mahasiswa",
                        "action": "Menyerahkan Form Cuti Akademik yang telah ditandatangani Ketua Program Studi ke Dekan; kemudian dilakukan disposisi ke Subbag Akademik dan Kemahasiswaan",
                        "inputs": [],
                        "output": "",
                        "evidence": "Mahasiswa menyerahkan Form Cuti Akademik yang telah ditandatangani Ketua Program Studi ke Dekan."
                    },
                    {
                        "step_num": 4,
                        "actor": "Subbag Akademik dan Kemahasiswaan",
                        "action": "Menerima, memeriksa, dan meneliti kelengkapan persyaratan form cuti akademik; jika lengkap, form diberi paraf dan diproses lebih lanjut",
                        "inputs": [],
                        "output": "",
                        "evidence": "Subbag Akademik dan Kemahasiswaan menerima, memeriksa, dan meneliti kelengkapan persyaratan form cuti akademik."
                    },
                    {
                        "step_num": 5,
                        "actor": "Dekan",
                        "action": "Menandatangani Surat Izin Dekan",
                        "inputs": [],
                        "output": "Surat Izin Dekan",
                        "evidence": "Dekan menandatangani Surat Izin Dekan."
                    },
                    {
                        "step_num": 6,
                        "actor": "Subbag Sumber Daya",
                        "action": "Memberi nomor surat dan mengirimkannya ke Subbag Akademik dan Kemahasiswaan dengan tembusan ke Dosen Wali dan Ketua Program Studi",
                        "inputs": [],
                        "output": "Surat Izin Dekan",
                        "evidence": "Subbag Sumber Daya memberi nomor surat dan mengirimkannya ke Subbag Akademik dan Kemahasiswaan dengan tembusan ke Dosen Wali dan Ketua Program Studi."
                    },
                    {
                        "step_num": 7,
                        "actor": "Subbag Akademik dan Kemahasiswaan",
                        "action": "Mengupdate status mahasiswa di Sistem Akademik, mengarsipkan Surat Izin Dekan, dan mengirimkan kepada mahasiswa",
                        "inputs": [],
                        "output": "Surat Izin Dekan",
                        "evidence": "Subbag Akademik dan Kemahasiswaan mengupdate status mahasiswa di Sistem Akademik, mengarsipkan Surat Izin Dekan, dan mengirimkan kepada mahasiswa."
                    },
                    {
                        "step_num": 8,
                        "actor": "Mahasiswa",
                        "action": "Mengambil Surat Izin Dekan di loket akademik",
                        "inputs": [],
                        "output": "Surat Izin Dekan",
                        "evidence": "Mahasiswa dapat mengambil Surat Izin Dekan di loket akademik."
                    }
                ]
            },
            {
                "id": "SOP_LEGALISIR",
                "title": "Legalisir Ijazah Dan Transkrip",
                "source_pdf": "SOP_Legalisir_Ijazah_Dan_Transkrip.pdf",
                "aliases": ["legalisir", "legalisir ijazah", "legalisir transkrip", "akreditasi program studi"],
                "max_duration": "3 hari kerja",
                "max_duration_evidence": "total maksimal waktu pemrosesan maksimal 3 (tiga) hari kerja",
                "steps": [
                    {
                        "step_num": 1,
                        "actor": "Alumni",
                        "action": "Menyerahkan foto copy ijazah/transkrip nilai/sertifikat akreditasi program studi beserta surat aslinya kepada petugas Subbag Akademik dan Kemahasiswaan",
                        "inputs": ["Berkas Pengajuan Legalisir"],
                        "output": "Berkas Pengajuan Legalisir",
                        "duration": "±5 menit",
                        "evidence": "Alumni menyerahkan foto copy ijazah/transkrip nilai/sertifikat akreditasi program studi beserta surat aslinya kepada petugas Subbag Akademik dan Kemahasiswaan."
                    },
                    {
                        "step_num": 2,
                        "actor": "Petugas Subbag Akademik dan Kemahasiswaan",
                        "action": "Memeriksa keabsahan, kesesuaian foto copy ijazah/transkrip nilai/sertifikat akreditasi program studi dengan surat aslinya dan memberikan cap",
                        "inputs": [],
                        "output": "Persetujuan Legalisir",
                        "duration": "±10 menit",
                        "evidence": "Petugas Subbag Akademik dan Kemahasiswaan memeriksa keabsahan, kesesuaian foto copy ijazah/transkrip nilai/sertifikat akreditasi program studi dengan surat aslinya dan memberikan cap."
                    },
                    {
                        "step_num": 3,
                        "actor": "Supervisor Akademik dan Kemahasiswaan",
                        "action": "Memeriksa dan memberi paraf di samping kanan nama Dekan",
                        "inputs": [],
                        "output": "Paraf Supervisor Akademik dan Kemahasiswaan",
                        "duration": "±15 menit",
                        "evidence": "Supervisor Akademik dan Kemahasiswaan memeriksa dan memberi paraf di samping kanan nama Dekan."
                    },
                    {
                        "step_num": 4,
                        "actor": "Dekan/Wakil Dekan Akademik dan Kemahasiswaan",
                        "action": "Menandatangani pada 'Pengesahan Telah diperiksa Kebenarannya dan Sesuai dengan Aslinya'",
                        "inputs": [],
                        "output": "TTD Dekan/Wakil Dekan Akademik dan Kemahasiswaan",
                        "duration": "±1 hari",
                        "evidence": "Dekan/Wakil Dekan Akademik dan Kemahasiswaan menandatangani pada"
                    },
                    {
                        "step_num": 5,
                        "actor": "Petugas Subbag Akademik dan Kemahasiswaan",
                        "action": "Mengambil kembali Legalisir yang sudah ditandatangani Dekan/Wakil Dekan Akademik dan Kemahasiswaan dan memberikan stempel Fakultas Sains dan Matematika",
                        "inputs": [],
                        "output": "Stempel Fakultas Sains dan Matematika",
                        "duration": "±5 menit",
                        "evidence": "Petugas Subbag Akademik dan Kemahasiswaan mengambil kembali Legalisir yang sudah ditanda tangani Dekan / Wakil Dekan Akademik dan Kemahasiswaan dan memberikan stempel Fakultas Sains dan Matematika."
                    },
                    {
                        "step_num": 6,
                        "actor": "Alumni",
                        "action": "Mengambil Legalisir di Subbag Akademik dan Kemahasiswaan dengan membawa aslinya, menulis di buku pengambilan, dan menandatanganinya",
                        "inputs": [],
                        "output": "Legalisir Ijazah/transkrip nilai/sertifikat akreditasi",
                        "duration": "±15 menit",
                        "evidence": "Alumni mengambil Legalisir di Subbag Akademik dan Kemahasiswaan dengan membawa aslinya dan menulis di buku pengambilan dan di tanda tangani ybs."
                    }
                ]
            },
            {
                "id": "SOP_PENGISIAN_IRS",
                "title": "Pengisian Isian Rencana Studi (IRS)",
                "source_pdf": "SOP_Pengisian_IRS.pdf",
                "aliases": ["pengisian irs", "isi irs", "rencana studi", "konsultasi dosen wali", "her-registrasi"],
                "max_duration": "3 hari kerja",
                "max_duration_evidence": "total maksimal waktu pemrosesan maksimal 3 (tiga) hari kerja",
                "steps": [
                    {
                        "step_num": 1,
                        "actor": "Mahasiswa",
                        "action": "Membayar biaya pendidikan pada bank yang bekerja sama dengan Undip (BNI, Mandiri, BTN, dan BRI)",
                        "inputs": [],
                        "output": "Bukti Pembayaran SPP/UKT",
                        "duration": "±10 menit",
                        "evidence": "Mahasiswa membayar biaya pendidikan pada bank yang bekerja sama dengan Undip (BNI, Mandiri, BTN, dan BRI)."
                    },
                    {
                        "step_num": 2,
                        "actor": "Mahasiswa",
                        "action": "Melakukan her-registrasi online melalui SIAP",
                        "inputs": [],
                        "output": "",
                        "duration": "±10 menit",
                        "evidence": "Mahasiswa melakukan her-registrasi online melalui SIAP."
                    },
                    {
                        "step_num": 3,
                        "actor": "Mahasiswa",
                        "action": "Melakukan pengisian, perbaikan, atau mencetak IRS sementara secara online melalui SIAP",
                        "inputs": [],
                        "output": "",
                        "duration": "±10 menit",
                        "evidence": "Mahasiswa melakukan pengisian, perbaikan, atau mencetak IRS sementara secara online melalui SIAP."
                    },
                    {
                        "step_num": 4,
                        "actor": "Mahasiswa",
                        "action": "Menemui Pembimbing Akademik untuk berkonsultasi mengenai mata kuliah yang akan diambil; Pembimbing Akademik memberikan arahan dengan mempertimbangkan ketentuan jumlah SKS maksimal pada semester terkait",
                        "inputs": [],
                        "output": "Print out IRS / Online IRS",
                        "duration": "±10 menit",
                        "evidence": "Mahasiswa menemui Pembimbing Akademik untuk berkonsultasi mengenai mata kuliah yang akan diambil."
                    },
                    {
                        "step_num": 5,
                        "actor": "Pembimbing Akademik",
                        "action": "Apabila mata kuliah dan jumlah SKS sudah sesuai, melakukan persetujuan secara online pada SIAP",
                        "inputs": [],
                        "output": "Persetujuan Pembimbing Akademik melalui SIAP",
                        "duration": "±10 menit",
                        "evidence": "Apabila mata kuliah dan jumlah SKS sudah sesuai, Pembimbing Akademik melakukan persetujuan secara online pada SIAP."
                    },
                    {
                        "step_num": 6,
                        "actor": "Mahasiswa",
                        "action": "Melakukan pengecekan IRS online di sistem SIAP bila diperlukan",
                        "inputs": [],
                        "output": "File IRS SIAP",
                        "duration": "±10 menit",
                        "evidence": "Mahasiswa dapat melakukan pengecekan IRS online di sistem SIAP bila diperlukan."
                    }
                ]
            },
            {
                "id": "SOP_KETERLAMBATAN_UKT",
                "title": "Permohonan Izin Keterlambatan Pembayaran UKT",
                "source_pdf": "SOP_Permohonan_Izin_Keterlambatan_Pembayaran_UKT.pdf",
                "aliases": ["keterlambatan ukt", "izin terlambat ukt", "dispensasi ukt", "tenggang ukt"],
                "max_duration": "3 hari kerja",
                "max_duration_evidence": "3 (tiga) hari kerja",
                "steps": [
                    {
                        "step_num": 1,
                        "actor": "Mahasiswa",
                        "action": "Mengunduh, mengisi, dan menandatangani Form Permohonan Keterlambatan Pembayaran UKT serta melampirkan persyaratan",
                        "inputs": [],
                        "output": "",
                        "link": "https://drive.google.com/file/d/1o7hogihjZsFTB_V27JU1OfNGBuI2Sr55/",
                        "evidence": "Mahasiswa mengunduh, mengisi, dan menandatangani Form Permohonan Keterlambatan Pembayaran UKT serta melampirkan persyaratan."
                    },
                    {
                        "step_num": 2,
                        "actor": "Mahasiswa",
                        "action": "Meminta tanda tangan dosen wali serta persetujuan dan tanda tangan Ketua Program Studi",
                        "inputs": [],
                        "output": "",
                        "evidence": "Mahasiswa meminta tanda tangan dosen wali serta persetujuan dan tanda tangan Ketua Program Studi."
                    },
                    {
                        "step_num": 3,
                        "actor": "Mahasiswa",
                        "action": "Menyerahkan Form Permohonan Keterlambatan UKT yang telah ditandatangani dan disetujui kepada Supervisor Sumber Daya",
                        "inputs": [],
                        "output": "",
                        "evidence": "Mahasiswa menyerahkan Form Permohonan Keterlambatan UKT yang telah ditandatangani dan disetujui kepada Supervisor Sumber Daya."
                    },
                    {
                        "step_num": 4,
                        "actor": "Supervisor Sumber Daya",
                        "action": "Menerima, memeriksa, dan meneliti kelengkapan persyaratan form, serta memberikan paraf",
                        "inputs": [],
                        "output": "",
                        "evidence": "Supervisor Sumber Daya menerima, memeriksa, dan meneliti kelengkapan persyaratan form, serta memberikan paraf."
                    },
                    {
                        "step_num": 5,
                        "actor": "Wakil Dekan Sumber Daya",
                        "action": "Memproses Surat Permohonan Izin Keterlambatan Pembayaran UKT",
                        "inputs": [],
                        "output": "",
                        "evidence": "Wakil Dekan Sumber Daya memproses Surat Permohonan Izin Keterlambatan Pembayaran UKT."
                    },
                    {
                        "step_num": 6,
                        "actor": "Wakil Dekan Sumber Daya",
                        "action": "Menandatangani Surat Permohonan Izin Keterlambatan Pembayaran UKT",
                        "inputs": [],
                        "output": "",
                        "evidence": "Wakil Dekan Sumber Daya menandatangani Surat Permohonan Izin Keterlambatan Pembayaran UKT."
                    },
                    {
                        "step_num": 7,
                        "actor": "Subbag Sumber Daya",
                        "action": "Memproses Surat Permohonan Keterlambatan Pembayaran UKT",
                        "inputs": [],
                        "output": "",
                        "evidence": "Subbag Sumber Daya memproses Surat Permohonan Keterlambatan Pembayaran UKT."
                    },
                    {
                        "step_num": 8,
                        "actor": "Mahasiswa",
                        "action": "Membawa Surat Permohonan Keterlambatan Pembayaran UKT kepada Wakil Rektor II melalui Manajer Akademik",
                        "inputs": [],
                        "output": "",
                        "evidence": "Mahasiswa membawa Surat Permohonan Keterlambatan Pembayaran UKT kepada Wakil Rektor II melalui Manajer Akademik."
                    },
                    {
                        "step_num": 9,
                        "actor": "Wakil Rektor II",
                        "action": "Mendisposisikan ke Direktorat Keuangan untuk memproses pembayaran UKT",
                        "inputs": [],
                        "output": "",
                        "evidence": "Wakil Rektor II mendisposisikan ke Direktorat Keuangan untuk memproses pembayaran UKT."
                    },
                    {
                        "step_num": 10,
                        "actor": "Bendahara Penerimaan UKT",
                        "action": "Membuka sistem pembayaran mahasiswa",
                        "inputs": [],
                        "output": "",
                        "evidence": "Bendahara Penerimaan UKT membuka sistem pembayaran mahasiswa."
                    },
                    {
                        "step_num": 11,
                        "actor": "Mahasiswa",
                        "action": "Membayar biaya UKT ke bank",
                        "inputs": [],
                        "output": "",
                        "evidence": "Mahasiswa membayar biaya UKT ke bank."
                    }
                ]
            },
            {
                "id": "SOP_AKTIF_SETELAH_CUTI",
                "title": "Permohonan Izin Aktif Setelah Cuti",
                "source_pdf": "SOP_Permohonan_Izin_Aktif_Setelah_Cuti.pdf",
                "aliases": ["aktif setelah cuti", "aktif kembali", "selesai cuti", "lapor aktif"],
                # The PDF states no overall processing limit for this SOP.
                "max_duration": None,
                "max_duration_evidence": None,
                "steps": [
                    {
                        "step_num": 1,
                        "actor": "Mahasiswa",
                        "action": "Membawa surat izin cuti akademik semester dan melapor kepada Ketua Program Studi dan Subbag Akademik dan Kemahasiswaan",
                        "inputs": ["Surat Izin Cuti", "Fotokopi Kartu Tanda Mahasiswa (KTM)"],
                        "output": "",
                        "duration": "±5 menit",
                        "evidence": "Mahasiswa membawa surat izin cuti akademik semester melapor kepada Ketua Program Studi dan Subbag Akademik dan Kemahasiswaan."
                    },
                    {
                        "step_num": 2,
                        "actor": "Mahasiswa",
                        "action": "Melakukan registrasi online pada SSO masing-masing mahasiswa",
                        "inputs": [],
                        "output": "",
                        "duration": "±5 menit",
                        "evidence": "Mahasiswa melakukan registrasi online pada SSO masing-masing mahasiswa."
                    },
                    {
                        "step_num": 3,
                        "actor": "Mahasiswa",
                        "action": "Terdaftar sebagai Peserta Kuliah / Mahasiswa Aktif di Fakultas",
                        "inputs": [],
                        "output": "",
                        "duration": "±5 menit",
                        "evidence": "Mahasiswa terdaftar sebagai Peserta Kuliah / Mahasiswa Aktif di Fakuktas."
                    }
                ]
            },
            {
                "id": "SOP_REKOMENDASI_BEASISWA",
                "title": "Surat Pengajuan Rekomendasi Beasiswa",
                "source_pdf": "SOP_Pengajuan_Rekomendasi_Beasiswa.pdf",
                "aliases": ["rekomendasi beasiswa", "surat beasiswa", "syarat beasiswa", "pengajuan beasiswa"],
                "max_duration": "3 hari, 45 menit waktu kerja",
                "max_duration_evidence": "3 (tiga) hari, 45 menit waktu kerja",
                "steps": [
                    {
                        "step_num": 1,
                        "actor": "Mahasiswa",
                        "action": "Mendownload dan mengisi formulir/Surat Rekomendasi Pengajuan Beasiswa ke Subbag Akademik dan Kemahasiswaan (BAK) Fakultas",
                        "inputs": ["Surat Rekomendasi Beasiswa", "KHS yang dilegalisir"],
                        "output": "Surat Rekomendasi Beasiswa dan KHS yang dilegalisir",
                        "duration": "±30 menit",
                        "link": "https://drive.google.com/file/d/18f_nbPXHbElgRNoplDFQVNYGGDkFxBTT/view?usp=sharing",
                        "evidence": "Mahasiswa mendownload dan mengisi formulir/ Surat Rekomendasi Pengajuan Beasiswa ke Subbag Akademik dan Kemahasiswaan (BAK) Fakultas."
                    },
                    {
                        "step_num": 2,
                        "actor": "BAK Fakultas",
                        "action": "Meneliti dan memverifikasi Surat Rekomendasi Pengajuan Beasiswa",
                        "inputs": [],
                        "output": "",
                        "duration": "±1 hari",
                        "evidence": "BAK Fakultas meneliti dan memverifikasi Surat Rekomendasi Pengajuan Beasiswa."
                    },
                    {
                        "step_num": 3,
                        "actor": "BAK Fakultas",
                        "action": "Memberikan paraf dan nomor surat pada Surat Rekomendasi Pengajuan Beasiswa tersebut",
                        "inputs": [],
                        "output": "",
                        "duration": "±10 menit",
                        "evidence": "BAK Fakultas memberikan paraf dan nomor surat pada Surat Rekomendasi Pengajuan Beasiswa tersebut."
                    },
                    {
                        "step_num": 4,
                        "actor": "Wakil Dekan Akademik dan Kemahasiswaan",
                        "action": "Memberikan tanda tangan pada Surat Rekomendasi Pengajuan Beasiswa tersebut",
                        "inputs": [],
                        "output": "",
                        "duration": "±2 hari",
                        "evidence": "Wakil Dekan Akademik dan Kemahasiswaan memberikan tanda tangan pada Surat Rekomendasi Pengajuan Beasiswa tersebut."
                    },
                    {
                        "step_num": 5,
                        "actor": "Mahasiswa",
                        "action": "Mengambil surat tersebut di BAK Fakultas dan mencatat bukti pengambilan",
                        "inputs": [],
                        "output": "",
                        "duration": "±5 menit",
                        "evidence": "Mahasiswa mengambil surat tersebut di BAK Fakultas dan mencatat bukti pengambilan."
                    }
                ]
            },
            {
                "id": "SOP_PROPOSAL_ORMAWA",
                "title": "Pengajuan Proposal Kegiatan Organisasi Mahasiswa",
                "source_pdf": "SOP_Pengajuan_Proposal_Kegiatan_Organisasi_Mahasiswa.pdf",
                "aliases": ["proposal ormawa", "proposal kegiatan", "organisasi mahasiswa", "izin kegiatan"],
                "max_duration": "3 hari kerja",
                "max_duration_evidence": "maksimal waktu pemrosesan maksimal 3 (tiga) hari kerja",
                "steps": [
                    {
                        "step_num": 1,
                        "actor": "Mahasiswa",
                        "action": "Menyerahkan proposal ke petugas kemahasiswaan dan Supervisor Akademik untuk registrasi dan alokasi ruang (jika kegiatan di FSM)",
                        "inputs": ["Proposal kegiatan organisasi"],
                        "output": "",
                        "evidence": "Mahasiswa menyerahkan proposal ke petugas kemahasiswaan dan Supervisor Akademik untuk registrasi dan alokasi ruang (jika kegiatan di FSM)."
                    },
                    {
                        "step_num": 2,
                        "actor": "Supervisor Akademik dan Kemahasiswaan",
                        "action": "Meneliti dan memverifikasi proposal kegiatan yang diajukan agar sesuai peraturan dan memberikan paraf",
                        "inputs": [],
                        "output": "Paraf Supervisor Akademik dan Kemahasiswaan",
                        "evidence": "Supervisor Akademik dan Kemahasiswaan meneliti dan memverifikasi proposal kegiatan yang diajukan agar sesuai peraturan dan memberikan paraf."
                    },
                    {
                        "step_num": 3,
                        "actor": "Wakil Dekan Akademik dan Kemahasiswaan",
                        "action": "Mengevaluasi substansi proposal kegiatan",
                        "inputs": [],
                        "output": "Persetujuan Wakil Dekan Akademik dan Kemahasiswaan",
                        "evidence": "Wakil Dekan Akademik dan Kemahasiswaan mengevaluasi substansi proposal kegiatan."
                    },
                    {
                        "step_num": 4,
                        "actor": "Wakil Dekan Akademik dan Kemahasiswaan",
                        "action": "Pengesahan oleh Wakil Dekan Akademik dan Kemahasiswaan",
                        "inputs": [],
                        "output": "",
                        "evidence": "Pengesahan Wakil Dekan Akademik dan Kemahasiswaan."
                    },
                    {
                        "step_num": 5,
                        "actor": "Mahasiswa",
                        "action": "Mengambil proposal yang telah disetujui",
                        "inputs": [],
                        "output": "Proposal kegiatan organisasi yang telah disetujui",
                        "evidence": "Mahasiswa mengambil proposal yang telah disetujui."
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
                source_pdf=sop["source_pdf"],
                max_duration=sop["max_duration"],
                max_duration_evidence=sop["max_duration_evidence"]
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
                    link=s.get("link", ""),
                    evidence=s["evidence"]
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

    def to_serializable(self) -> Dict[str, Any]:
        """
        Deterministic, JSON-serialisable view of the graph: SOP definitions plus
        every node and edge (sorted), so two graphs with the same content always
        serialise to the same string.
        """
        nodes = [[n, dict(sorted(attrs.items()))] for n, attrs in sorted(self.graph.nodes(data=True))]
        edges = [[u, v, dict(sorted(attrs.items()))] for u, v, attrs in sorted(self.graph.edges(data=True))]
        return {
            "sops": [self.sop_metadata[k] for k in sorted(self.sop_metadata)],
            "nodes": nodes,
            "edges": edges,
        }

    def graph_sha256(self) -> str:
        """SHA-256 of the canonical JSON serialisation (fingerprint for run_config.json)."""
        canonical = json.dumps(self.to_serializable(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

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

        finish_label = f'✅ Selesai ({sop["max_duration"]})' if sop.get("max_duration") else "✅ Selesai"
        mermaid.append(f'    Finish(["{finish_label}"])')
        
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
            
        finish_label = f'✅ Selesai ({sop["max_duration"]})' if sop.get("max_duration") else "✅ Selesai"
        mermaid.append(f'    Finish(["{finish_label}"])')
        
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
        limit = sop.get("max_duration") or "tidak disebutkan dalam SOP"
        mermaid.append(f'    Start(["🚀 Alur Lengkap: {sop["title"]}<br><i>Total Batas Waktu: {limit}</i>"])')
        
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
        lines.append(f"Maksimal Waktu Pemrosesan: {sop['max_duration'] or 'tidak disebutkan dalam dokumen SOP'}")
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
