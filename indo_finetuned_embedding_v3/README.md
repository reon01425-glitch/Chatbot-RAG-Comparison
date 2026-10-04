---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:50
- loss:MultipleNegativesRankingLoss
base_model: LazarusNLP/all-indo-e5-small-v4
widget:
- source_sentence: Siapa pihak program studi yang dimintai persetujuan berkas cuti
    oleh mahasiswa?
  sentences:
  - Waktu untuk mendownload dan mengisi formulir/Surat Rekomendasi Pengajuan Beasiswa
    adalah ±30 menit.
  - Mahasiswa meminta persetujuan dan tanda tangan kepada Ketua Program Studi.
  - Mahasiswa melakukan pengisian, perbaikan, atau mencetak IRS sementara secara online
    melalui SIAP.
- source_sentence: Unit kerja mana di fakultas yang menyelesaikan pemrosesan administratif
    surat keterlambatan UKT?
  sentences:
  - Subbag Sumber Daya memproses Surat Permohonan Keterlambatan Pembayaran UKT.
  - Mahasiswa terdaftar sebagai Peserta Kuliah / Mahasiswa Aktif di Fakuktas.
  - Pembimbing memberikan arahan dengan mempertimbangkan ketentuan jumlah SKS maksimal
    pada semester terkait.
- source_sentence: Apa tanda administratif yang dibubuhkan staf BAK pada draf surat
    rekomendasi beasiswa?
  sentences:
  - BAK Fakultas memberikan paraf dan nomor surat pada Surat Rekomendasi Pengajuan
    Beasiswa tersebut.
  - Wakil Dekan Akademik dan Kemahasiswaan mengevaluasi substansi proposal kegiatan.
  - Supervisor Akademik dan Kemahasiswaan memeriksa dan memberi paraf di samping kanan
    nama Dekan.
- source_sentence: Cap stempel apa yang dibubuhkan petugas setelah berkas legalisir
    ditandatangani Dekan?
  sentences:
  - Jika berkas persyaratan lengkap, form diberi paraf dan diproses lebih lanjut.
  - Setelah berkas cuti diserahkan ke Dekan, kemudian dilakukan disposisi ke Subbag
    Akademik dan Kemahasiswaan.
  - Petugas Subbag Akademik dan Kemahasiswaan memberikan stempel Fakultas Sains dan
    Matematika.
- source_sentence: Tindakan awal apa yang harus dikerjakan mahasiswa saat menyusun
    permohonan keterlambatan UKT?
  sentences:
  - Pengesahan Wakil Dekan Akademik dan Kemahasiswaan.
  - Wakil Rektor II mendisposisikan ke Direktorat Keuangan untuk memproses pembayaran
    UKT.
  - Mahasiswa mengunduh, mengisi, dan menandatangani Form Permohonan Keterlambatan
    Pembayaran UKT serta melampirkan persyaratan.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on LazarusNLP/all-indo-e5-small-v4

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [LazarusNLP/all-indo-e5-small-v4](https://huggingface.co/LazarusNLP/all-indo-e5-small-v4). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [LazarusNLP/all-indo-e5-small-v4](https://huggingface.co/LazarusNLP/all-indo-e5-small-v4) <!-- at revision ac1d8e6328f8167079ea3b2ba078583f77a2e40a -->
- **Maximum Sequence Length:** 128 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Tindakan awal apa yang harus dikerjakan mahasiswa saat menyusun permohonan keterlambatan UKT?',
    'Mahasiswa mengunduh, mengisi, dan menandatangani Form Permohonan Keterlambatan Pembayaran UKT serta melampirkan persyaratan.',
    'Wakil Rektor II mendisposisikan ke Direktorat Keuangan untuk memproses pembayaran UKT.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6314, 0.0891],
#         [0.6314, 1.0000, 0.1940],
#         [0.0891, 0.1940, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 50 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 50 samples:
  |         | sentence_0                                                                         | sentence_1                                                                         | label                                                         |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | float                                                         |
  | details | <ul><li>min: 16 tokens</li><li>mean: 21.06 tokens</li><li>max: 27 tokens</li></ul> | <ul><li>min: 10 tokens</li><li>mean: 22.28 tokens</li><li>max: 43 tokens</li></ul> | <ul><li>min: 1.0</li><li>mean: 1.0</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                  | sentence_1                                                                                                     | label            |
  |:----------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Status kemahasiswaan apa yang diperoleh setelah seluruh alur lapor aktif pasca-cuti terpenuhi?</code>                 | <code>Mahasiswa terdaftar sebagai Peserta Kuliah / Mahasiswa Aktif di Fakuktas.</code>                         | <code>1.0</code> |
  | <code>Bagaimana mekanisme persetujuan IRS oleh Pembimbing Akademik apabila susunan mata kuliah telah sesuai?</code>         | <code>Pembimbing Akademik melakukan persetujuan secara online pada SIAP.</code>                                | <code>1.0</code> |
  | <code>Siapa dua pihak akademik di tingkat departemen yang wajib membubuhkan persetujuan pada form keterlambatan UKT?</code> | <code>Mahasiswa meminta tanda tangan dosen wali serta persetujuan dan tanda tangan Ketua Program Studi.</code> | <code>1.0</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `num_train_epochs`: 10
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 8
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 10
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Framework Versions
- Python: 3.12.5
- Sentence Transformers: 5.1.1
- Transformers: 4.56.2
- PyTorch: 2.8.0
- Accelerate: 1.12.0
- Datasets: 4.6.1
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->