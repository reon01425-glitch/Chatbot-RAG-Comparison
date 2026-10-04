import json
import os

DATASETS = {
    "train_SOP_Izin_Cuti_Akademik.json": [
        {
            "id": "data/SOP_Izin_Cuti_Akademik.pdf:0:0",
            "qa": "Q: Berapa batas waktu pengerjaan pengurusan izin cuti akademik bagi mahasiswa FSM?\nA: Total maksimal waktu pemrosesan izin cuti akademik adalah 3 (tiga) hari kerja.",
            "source": "data/SOP_Izin_Cuti_Akademik.pdf"
        },
        {
            "id": "data/SOP_Izin_Cuti_Akademik.pdf:0:1",
            "qa": "Q: Sistem daring apa yang wajib diisi mahasiswa untuk pengajuan cuti akademik?\nA: Mahasiswa wajib mengisi pengajuan secara online melalui sistem SIAP serta melampirkan berkas persyaratan pada formulir cuti.",
            "source": "data/SOP_Izin_Cuti_Akademik.pdf"
        },
        {
            "id": "data/SOP_Izin_Cuti_Akademik.pdf:0:2",
            "qa": "Q: Siapa pihak program studi yang dimintai persetujuan berkas cuti oleh mahasiswa?\nA: Mahasiswa meminta persetujuan dan tanda tangan kepada Ketua Program Studi.",
            "source": "data/SOP_Izin_Cuti_Akademik.pdf"
        },
        {
            "id": "data/SOP_Izin_Cuti_Akademik.pdf:0:3",
            "qa": "Q: Ke mana form cuti yang telah disetujui Kaprodi diserahkan oleh pemohon?\nA: Formulir diserahkan ke Dekan, kemudian dilakukan disposisi ke Subbag Akademik dan Kemahasiswaan.",
            "source": "data/SOP_Izin_Cuti_Akademik.pdf"
        },
        {
            "id": "data/SOP_Izin_Cuti_Akademik.pdf:0:4",
            "qa": "Q: Tindakan apa yang dilakukan staf Subbag Akademik setelah menerima disposisi permohonan cuti?\nA: Petugas Subbag Akademik dan Kemahasiswaan memverifikasi berkas persyaratan dan menyusun konsep Surat Izin Cuti Akademik.",
            "source": "data/SOP_Izin_Cuti_Akademik.pdf"
        },
        {
            "id": "data/SOP_Izin_Cuti_Akademik.pdf:0:5",
            "qa": "Q: Siapa pejabat tata usaha yang memeriksa draf surat izin cuti sebelum diajukan ke pimpinan fakultas?\nA: Supervisor Subbag Akademik dan Kemahasiswaan serta Manajer Bagian Tata Usaha memeriksa dan memberi paraf konsep surat.",
            "source": "data/SOP_Izin_Cuti_Akademik.pdf"
        },
        {
            "id": "data/SOP_Izin_Cuti_Akademik.pdf:0:6",
            "qa": "Q: Siapa saja pihak yang menerima salinan tembusan Surat Izin Dekan terkait cuti akademik?\nA: Surat Izin Dekan diberikan dengan tembusan kepada Dosen Wali dan Ketua Program Studi mahasiswa.",
            "source": "data/SOP_Izin_Cuti_Akademik.pdf"
        },
        {
            "id": "data/SOP_Izin_Cuti_Akademik.pdf:0:7",
            "qa": "Q: Langkah administratif apa yang dilakukan Subbag Akademik setelah Surat Izin Dekan terbit?\nA: Subbag Akademik dan Kemahasiswaan memperbarui status mahasiswa di Sistem Akademik serta mengarsipkan surat izin.",
            "source": "data/SOP_Izin_Cuti_Akademik.pdf"
        },
        {
            "id": "data/SOP_Izin_Cuti_Akademik.pdf:0:8",
            "qa": "Q: Di loket mana mahasiswa dapat mengambil fisik surat izin cuti yang telah selesai diproses?\nA: Mahasiswa dapat mengambil fisik Surat Izin Dekan di loket akademik fakultas.",
            "source": "data/SOP_Izin_Cuti_Akademik.pdf"
        }
    ],
    "train_SOP_Legalisir_Ijazah_Dan_Transkrip.json": [
        {
            "id": "data/SOP_Legalisir_Ijazah_Dan_Transkrip.pdf:0:0",
            "qa": "Q: Berapa estimasi waktu maksimal pemrosesan pengesahan legalisir di lingkungan FSM Undip?\nA: Total maksimal durasi pemrosesan pengesahan legalisir ijazah dan transkrip adalah 3 (tiga) hari kerja.",
            "source": "data/SOP_Legalisir_Ijazah_Dan_Transkrip.pdf"
        },
        {
            "id": "data/SOP_Legalisir_Ijazah_Dan_Transkrip.pdf:0:1",
            "qa": "Q: Berkas kelengkapan apa yang diserahkan alumni saat mengajukan pengesahan salinan ijazah?\nA: Alumni menyerahkan foto copy ijazah, transkrip nilai, atau sertifikat akreditasi beserta dokumen aslinya kepada Subbag Akademik dan Kemahasiswaan.",
            "source": "data/SOP_Legalisir_Ijazah_Dan_Transkrip.pdf"
        },
        {
            "id": "data/SOP_Legalisir_Ijazah_Dan_Transkrip.pdf:0:2",
            "qa": "Q: Apa tindakan petugas Subbag Akademik dalam memvalidasi salinan transkrip yang diajukan alumni?\nA: Petugas memeriksa keabsahan dan kesesuaian dokumen fotokopi dengan surat aslinya lalu memberikan cap.",
            "source": "data/SOP_Legalisir_Ijazah_Dan_Transkrip.pdf"
        },
        {
            "id": "data/SOP_Legalisir_Ijazah_Dan_Transkrip.pdf:0:3",
            "qa": "Q: Di sebelah bagian mana Supervisor Akademik membubuhkan paraf verifikasi pada dokumen legalisir?\nA: Supervisor Akademik dan Kemahasiswaan memeriksa berkas dan membubuhkan paraf di samping kanan nama Dekan.",
            "source": "data/SOP_Legalisir_Ijazah_Dan_Transkrip.pdf"
        },
        {
            "id": "data/SOP_Legalisir_Ijazah_Dan_Transkrip.pdf:0:4",
            "qa": "Q: Rumusan kalimat apa yang tercantum pada pengesahan berkas yang ditandatangani Dekan atau Wakil Dekan?\nA: Dekan atau Wakil Dekan Akademik dan Kemahasiswaan menandatangani pernyataan 'Pengesahan Telah diperiksa Kebenarannya dan Sesuai dengan Aslinya'.",
            "source": "data/SOP_Legalisir_Ijazah_Dan_Transkrip.pdf"
        },
        {
            "id": "data/SOP_Legalisir_Ijazah_Dan_Transkrip.pdf:0:5",
            "qa": "Q: Cap stempel apa yang dibubuhkan petugas setelah berkas legalisir ditandatangani Dekan?\nA: Petugas Subbag Akademik dan Kemahasiswaan memberikan stempel resmi Fakultas Sains dan Matematika pada berkas legalisir.",
            "source": "data/SOP_Legalisir_Ijazah_Dan_Transkrip.pdf"
        },
        {
            "id": "data/SOP_Legalisir_Ijazah_Dan_Transkrip.pdf:0:6",
            "qa": "Q: Prosedur apa yang harus dipenuhi pemohon saat mengambil berkas legalisir yang telah rampung?\nA: Alumni menunjukkan dokumen asli dan mencatatkan tanda tangan pada buku pengambilan di Subbag Akademik dan Kemahasiswaan.",
            "source": "data/SOP_Legalisir_Ijazah_Dan_Transkrip.pdf"
        }
    ],
    "train_SOP_Pengajuan_Proposal_Kegiatan_Organisasi_Mahasiswa.json": [
        {
            "id": "data/SOP_Pengajuan_Proposal_Kegiatan_Organisasi_Mahasiswa.pdf:0:0",
            "qa": "Q: Berapa alokasi batas waktu penyelesaian pengajuan usulan kegiatan ormawa di fakultas?\nA: Total durasi maksimal pemrosesan usulan proposal kegiatan organisasi mahasiswa adalah 3 (tiga) hari kerja.",
            "source": "data/SOP_Pengajuan_Proposal_Kegiatan_Organisasi_Mahasiswa.pdf"
        },
        {
            "id": "data/SOP_Pengajuan_Proposal_Kegiatan_Organisasi_Mahasiswa.pdf:0:1",
            "qa": "Q: Kapan alokasi ruangan diproses saat penyerahan proposal kegiatan himpunan atau ormawa?\nA: Alokasi ruang diproses jika kegiatan ormawa bertempat di lingkungan kampus FSM saat penyerahan proposal ke petugas dan Supervisor Akademik.",
            "source": "data/SOP_Pengajuan_Proposal_Kegiatan_Organisasi_Mahasiswa.pdf"
        },
        {
            "id": "data/SOP_Pengajuan_Proposal_Kegiatan_Organisasi_Mahasiswa.pdf:0:2",
            "qa": "Q: Apa fokus verifikasi yang dijalankan Supervisor Akademik terhadap proposal kemahasiswaan?\nA: Supervisor Akademik dan Kemahasiswaan meneliti kesesuaian proposal kegiatan dengan ketentuan peraturan yang berlaku dan memberi paraf.",
            "source": "data/SOP_Pengajuan_Proposal_Kegiatan_Organisasi_Mahasiswa.pdf"
        },
        {
            "id": "data/SOP_Pengajuan_Proposal_Kegiatan_Organisasi_Mahasiswa.pdf:0:3",
            "qa": "Q: Pejabat pimpinan mana yang bertanggung jawab menelaah kelayakan materi proposal kegiatan ormawa?\nA: Wakil Dekan Akademik dan Kemahasiswaan mengevaluasi substansi materi proposal kegiatan ormawa.",
            "source": "data/SOP_Pengajuan_Proposal_Kegiatan_Organisasi_Mahasiswa.pdf"
        },
        {
            "id": "data/SOP_Pengajuan_Proposal_Kegiatan_Organisasi_Mahasiswa.pdf:0:4",
            "qa": "Q: Langkah validasi akhir apa yang dilakukan oleh pimpinan fakultas pada lembar usulan ormawa?\nA: Wakil Dekan Akademik dan Kemahasiswaan memberikan pengesahan formal pada proposal kegiatan.",
            "source": "data/SOP_Pengajuan_Proposal_Kegiatan_Organisasi_Mahasiswa.pdf"
        },
        {
            "id": "data/SOP_Pengajuan_Proposal_Kegiatan_Organisasi_Mahasiswa.pdf:0:5",
            "qa": "Q: Bagaimana tahapan akhir penyerahan berkas proposal ormawa yang telah divalidasi pimpinan?\nA: Mahasiswa mengambil kembali berkas fisik proposal kegiatan organisasi yang telah disetujui.",
            "source": "data/SOP_Pengajuan_Proposal_Kegiatan_Organisasi_Mahasiswa.pdf"
        }
    ],
    "train_SOP_Pengajuan_Rekomendasi_Beasiswa.json": [
        {
            "id": "data/SOP_Pengajuan_Rekomendasi_Beasiswa.pdf:0:0",
            "qa": "Q: Berapa rincian waktu kerja yang ditetapkan untuk menyelesaikan permohonan surat rekomendasi beasiswa?\nA: Waktu pemrosesan rekomendasi beasiswa membutuhkan waktu maksimal 3 hari 45 menit waktu kerja.",
            "source": "data/SOP_Pengajuan_Rekomendasi_Beasiswa.pdf"
        },
        {
            "id": "data/SOP_Pengajuan_Rekomendasi_Beasiswa.pdf:0:1",
            "qa": "Q: Formulir apa dan berkas nilai apa yang wajib disiapkan pemohon rekomendasi beasiswa fakultas?\nA: Mahasiswa mengunduh dan melengkapi Formulir Surat Rekomendasi Pengajuan Beasiswa serta menyertakan salinan KHS yang telah dilegalisir.",
            "source": "data/SOP_Pengajuan_Rekomendasi_Beasiswa.pdf"
        },
        {
            "id": "data/SOP_Pengajuan_Rekomendasi_Beasiswa.pdf:0:2",
            "qa": "Q: Berapa alokasi durasi verifikasi berkas permohonan beasiswa oleh staf BAK Fakultas?\nA: Pemeriksaan dan verifikasi berkas pengajuan beasiswa oleh BAK Fakultas berlangsung sekitar 1 hari kerja.",
            "source": "data/SOP_Pengajuan_Rekomendasi_Beasiswa.pdf"
        },
        {
            "id": "data/SOP_Pengajuan_Rekomendasi_Beasiswa.pdf:0:3",
            "qa": "Q: Apa tanda administratif yang dibubuhkan staf BAK pada draf surat rekomendasi beasiswa?\nA: BAK Fakultas membubuhkan paraf pemeriksaan serta memberikan nomor agenda surat resmi.",
            "source": "data/SOP_Pengajuan_Rekomendasi_Beasiswa.pdf"
        },
        {
            "id": "data/SOP_Pengajuan_Rekomendasi_Beasiswa.pdf:0:4",
            "qa": "Q: Berapa lama estimasi tahapan penandatanganan rekomendasi beasiswa oleh pimpinan fakultas?\nA: Tahap penandatanganan oleh Wakil Dekan Akademik dan Kemahasiswaan dialokasikan sekitar 2 hari kerja.",
            "source": "data/SOP_Pengajuan_Rekomendasi_Beasiswa.pdf"
        },
        {
            "id": "data/SOP_Pengajuan_Rekomendasi_Beasiswa.pdf:0:5",
            "qa": "Q: Prosedur apa yang wajib dilakukan mahasiswa pada saat menerima surat rekomendasi beasiswa dari BAK?\nA: Mahasiswa mengambil dokumen fisik di bagian BAK Fakultas dan mencatat bukti penerimaan berkas.",
            "source": "data/SOP_Pengajuan_Rekomendasi_Beasiswa.pdf"
        }
    ],
    "train_SOP_Pengisian_IRS.json": [
        {
            "id": "data/SOP_Pengisian_IRS.pdf:0:0",
            "qa": "Q: Berapa hari rentang waktu maksimal yang dialokasikan untuk menyelesaikan seluruh rangkaian IRS mahasiswa?\nA: Total maksimal durasi pemrosesan pengisian Isian Rencana Studi (IRS) adalah 3 (tiga) hari kerja.",
            "source": "data/SOP_Pengisian_IRS.pdf"
        },
        {
            "id": "data/SOP_Pengisian_IRS.pdf:0:1",
            "qa": "Q: Apa bukti setor perbankan yang diperoleh mahasiswa setelah menyelesaikan kewajiban biaya semester?\nA: Mahasiswa mendapatkan lembar bukti pembayaran SPP atau UKT dari teller atau kanal pembayaran bank mitra Undip.",
            "source": "data/SOP_Pengisian_IRS.pdf"
        },
        {
            "id": "data/SOP_Pengisian_IRS.pdf:0:2",
            "qa": "Q: Apa tahap registrasi ulang yang wajib diselesaikan setelah melunasi biaya pendidikan semesteran?\nA: Mahasiswa wajib menjalankan her-registrasi secara online melalui portal sistem informasi akademik SIAP.",
            "source": "data/SOP_Pengisian_IRS.pdf"
        },
        {
            "id": "data/SOP_Pengisian_IRS.pdf:0:3",
            "qa": "Q: Aktivitas penyusunan jadwal kuliah apa saja yang dapat dilakukan mahasiswa pada aplikasi SIAP?\nA: Mahasiswa melakukan pengisian mata kuliah, perbaikan rencana studi, atau mencetak draf IRS sementara secara daring di SIAP.",
            "source": "data/SOP_Pengisian_IRS.pdf"
        },
        {
            "id": "data/SOP_Pengisian_IRS.pdf:0:4",
            "qa": "Q: Aspek apa yang dievaluasi Dosen Pembimbing Akademik saat sesi bimbingan rencana studi mahasiswa?\nA: Pembimbing Akademik memberikan konsultasi pemilihan mata kuliah dengan mempertimbangkan pagu beban SKS maksimal semester berjalan.",
            "source": "data/SOP_Pengisian_IRS.pdf"
        },
        {
            "id": "data/SOP_Pengisian_IRS.pdf:0:5",
            "qa": "Q: Bagaimana mekanisme persetujuan IRS oleh dosen wali apabila susunan mata kuliah telah dinyatakan valid?\nA: Pembimbing Akademik menyetujui rencana studi secara digital langsung pada sistem SIAP.",
            "source": "data/SOP_Pengisian_IRS.pdf"
        },
        {
            "id": "data/SOP_Pengisian_IRS.pdf:0:6",
            "qa": "Q: Bagaimana cara mahasiswa memastikan bahwa IRS semesterannya telah sah dan terekam di pangkalan data?\nA: Mahasiswa dapat memantau status IRS online dan mengunduh berkas IRS pada portal SIAP.",
            "source": "data/SOP_Pengisian_IRS.pdf"
        }
    ],
    "train_SOP_Permohonan_Izin_Aktif_Setelah_Cuti.json": [
        {
            "id": "data/SOP_Permohonan_Izin_Aktif_Setelah_Cuti.pdf:0:0",
            "qa": "Q: Apa tujuan utama dari diterbitkannya prosedur resmi permohonan aktif kembali setelah cuti studi?\nA: SOP ini mengatur tata cara bagi mahasiswa yang selesai masa cuti akademik agar memperoleh izin aktif kembali sebagai peserta kuliah.",
            "source": "data/SOP_Permohonan_Izin_Aktif_Setelah_Cuti.pdf"
        },
        {
            "id": "data/SOP_Permohonan_Izin_Aktif_Setelah_Cuti.pdf:0:1",
            "qa": "Q: Berkas identitas diri apa yang diserahkan bersama lembar persetujuan libur studi sebelumnya kepada Kaprodi?\nA: Mahasiswa melampirkan fotokopi Kartu Tanda Mahasiswa (KTM) bersama surat izin cuti akademik sebelumnya.",
            "source": "data/SOP_Permohonan_Izin_Aktif_Setelah_Cuti.pdf"
        },
        {
            "id": "data/SOP_Permohonan_Izin_Aktif_Setelah_Cuti.pdf:0:2",
            "qa": "Q: Kanal portal tunggal apa yang diakses mahasiswa untuk aktivasi status perkuliahan secara mandiri?\nA: Mahasiswa mengakses akun Single Sign-On (SSO) masing-masing untuk melakukan pendaftaran online.",
            "source": "data/SOP_Permohonan_Izin_Aktif_Setelah_Cuti.pdf"
        },
        {
            "id": "data/SOP_Permohonan_Izin_Aktif_Setelah_Cuti.pdf:0:3",
            "qa": "Q: Status kemahasiswaan apa yang diperoleh setelah seluruh alur lapor aktif pasca-cuti terpenuhi?\nA: Mahasiswa resmi terdaftar kembali sebagai Peserta Kuliah dan Mahasiswa Aktif di lingkungan Fakultas Sains dan Matematika.",
            "source": "data/SOP_Permohonan_Izin_Aktif_Setelah_Cuti.pdf"
        }
    ],
    "train_SOP_Permohonan_Izin_Keterlambatan_Pembayaran_UKT.json": [
        {
            "id": "data/SOP_Permohonan_Izin_Keterlambatan_Pembayaran_UKT.pdf:0:0",
            "qa": "Q: Berapa alokasi masa kerja yang disediakan fakultas untuk memproses izin dispensasi keterlambatan UKT?\nA: Total batas waktu pemrosesan permohonan dispensasi keterlambatan UKT adalah maksimal 3 (tiga) hari kerja.",
            "source": "data/SOP_Permohonan_Izin_Keterlambatan_Pembayaran_UKT.pdf"
        },
        {
            "id": "data/SOP_Permohonan_Izin_Keterlambatan_Pembayaran_UKT.pdf:0:1",
            "qa": "Q: Tindakan awal apa yang harus dikerjakan mahasiswa saat menyusun permohonan dispensasi UKT?\nA: Mahasiswa mengunduh formulir resmi, mengisinya, menandatangani, serta melampirkan dokumen syarat keterlambatan.",
            "source": "data/SOP_Permohonan_Izin_Keterlambatan_Pembayaran_UKT.pdf"
        },
        {
            "id": "data/SOP_Permohonan_Izin_Keterlambatan_Pembayaran_UKT.pdf:0:2",
            "qa": "Q: Siapa dua pihak akademik di tingkat departemen yang wajib membubuhkan persetujuan pada form dispensasi UKT?\nA: Mahasiswa meminta tanda tangan persetujuan kepada Dosen Wali dan Ketua Program Studi.",
            "source": "data/SOP_Permohonan_Izin_Keterlambatan_Pembayaran_UKT.pdf"
        },
        {
            "id": "data/SOP_Permohonan_Izin_Keterlambatan_Pembayaran_UKT.pdf:0:3",
            "qa": "Q: Kepada staf pengelola sarana mana berkas dispensasi UKT diserahkan setelah ditandatangani departemen?\nA: Mahasiswa menyerahkan berkas permohonan keterlambatan yang telah lengkap kepada Supervisor Sumber Daya.",
            "source": "data/SOP_Permohonan_Izin_Keterlambatan_Pembayaran_UKT.pdf"
        },
        {
            "id": "data/SOP_Permohonan_Izin_Keterlambatan_Pembayaran_UKT.pdf:0:4",
            "qa": "Q: Pimpinan dekanat mana yang menangani permohonan dispensasi keterlambatan biaya kuliah di fakultas?\nA: Wakil Dekan Sumber Daya memproses penerbitan Surat Permohonan Izin Keterlambatan Pembayaran UKT.",
            "source": "data/SOP_Permohonan_Izin_Keterlambatan_Pembayaran_UKT.pdf"
        },
        {
            "id": "data/SOP_Permohonan_Izin_Keterlambatan_Pembayaran_UKT.pdf:0:5",
            "qa": "Q: Siapa pejabat berwenang di FSM yang menandatangani surat rekomendasi keterlambatan UKT untuk universitas?\nA: Wakil Dekan Sumber Daya menandatangani Surat Permohonan Izin Keterlambatan Pembayaran UKT.",
            "source": "data/SOP_Permohonan_Izin_Keterlambatan_Pembayaran_UKT.pdf"
        },
        {
            "id": "data/SOP_Permohonan_Izin_Keterlambatan_Pembayaran_UKT.pdf:0:6",
            "qa": "Q: Unit kerja mana di fakultas yang menyelesaikan pemrosesan administratif surat dispensasi UKT?\nA: Subbag Sumber Daya memproses penomoran dan administrasi Surat Permohonan Keterlambatan Pembayaran UKT.",
            "source": "data/SOP_Permohonan_Izin_Keterlambatan_Pembayaran_UKT.pdf"
        },
        {
            "id": "data/SOP_Permohonan_Izin_Keterlambatan_Pembayaran_UKT.pdf:0:7",
            "qa": "Q: Lewat perantara manajer apa berkas izin dispensasi UKT diserahkan ke tingkat universitas?\nA: Mahasiswa membawa surat permohonan kepada Wakil Rektor II melalui Manajer Akademik universitas.",
            "source": "data/SOP_Permohonan_Izin_Keterlambatan_Pembayaran_UKT.pdf"
        },
        {
            "id": "data/SOP_Permohonan_Izin_Keterlambatan_Pembayaran_UKT.pdf:0:8",
            "qa": "Q: Ke direktorat mana pimpinan universitas mendisposisikan surat permohonan dispensasi pembayaran UKT?\nA: Wakil Rektor II mendisposisikan surat permohonan kepada Direktorat Keuangan universitas.",
            "source": "data/SOP_Permohonan_Izin_Keterlambatan_Pembayaran_UKT.pdf"
        },
        {
            "id": "data/SOP_Permohonan_Izin_Keterlambatan_Pembayaran_UKT.pdf:0:9",
            "qa": "Q: Petugas keuangan mana yang memiliki otorisasi untuk membuka kembali sistem billing tagihan mahasiswa?\nA: Bendahara Penerimaan UKT universitas membuka portal tagihan sistem pembayaran mahasiswa yang bersangkutan.",
            "source": "data/SOP_Permohonan_Izin_Keterlambatan_Pembayaran_UKT.pdf"
        },
        {
            "id": "data/SOP_Permohonan_Izin_Keterlambatan_Pembayaran_UKT.pdf:0:10",
            "qa": "Q: Tindakan penutup apa yang dilakukan mahasiswa setelah portal billing UKT dibuka kembali?\nA: Mahasiswa menyetorkan biaya pembayaran UKT ke bank mitra yang ditunjuk.",
            "source": "data/SOP_Permohonan_Izin_Keterlambatan_Pembayaran_UKT.pdf"
        }
    ]
}

os.makedirs("datasets", exist_ok=True)
total_pairs = 0
for filename, items in DATASETS.items():
    filepath = os.path.join("datasets", filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"Created {filename} with {len(items)} QA pairs.")
    total_pairs += len(items)

print(f"Total QA pairs across all 7 SOPs: {total_pairs}")
