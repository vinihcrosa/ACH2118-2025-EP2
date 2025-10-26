from typing import Iterable, List, Sequence, Union, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
import time


class BertVectorizer:
    """
    Vetorizador baseado em BERT/HuggingFace Transformers.
    Não treina no corpus; apenas gera embeddings (pooling médio) dos tokens.

    Parâmetros
    ----------
    model_name : str
        Nome do modelo no HuggingFace Hub (ex.: 'neuralmind/bert-base-portuguese-cased').
    max_length : int
        Tamanho máximo de tokens por texto (com truncation/padding).
    batch_size : int
        Tamanho do lote para inferência.
    device : str
        'cpu' (padrão) ou 'cuda' caso disponível.
    """

    def __init__(
        self,
        model_name: str = "neuralmind/bert-base-portuguese-cased",
        max_length: int = 256,
        batch_size: int = 16,
        device: str = "cpu",
        verbose: bool = False,
        progress_every: int = 10,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        # Seleção automática de device se solicitado
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.verbose = verbose
        self.progress_every = max(1, int(progress_every))

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        self.output_size = int(getattr(self.model.config, "hidden_size", 768))

    def fit(self, texts: Union[pd.Series, Iterable[str], np.ndarray]) -> "BertVectorizer":
        # Sem ajuste necessário; retorna self para compatibilidade
        return self

    def fit_transform(self, texts: Union[pd.Series, Iterable[str], np.ndarray]) -> pd.DataFrame:
        return self.fit(texts).transform(texts)

    def transform(self, texts: Union[pd.Series, Iterable[str], np.ndarray]) -> pd.DataFrame:
        corpus = self._prepare_texts(texts)
        vectors = self._embed_batches(corpus)
        columns = [f"bert_{i}" for i in range(self.output_size)]
        return pd.DataFrame(vectors, columns=columns)

    def _prepare_texts(self, texts: Union[pd.Series, Iterable[str], np.ndarray]) -> List[str]:
        if isinstance(texts, pd.Series):
            iterable = texts.fillna("").astype(str).tolist()
        elif isinstance(texts, np.ndarray):
            iterable = ["" if v is None else str(v) for v in texts.tolist()]
        elif isinstance(texts, Iterable) and not isinstance(texts, (str, bytes)):
            iterable = ["" if v is None else str(v) for v in texts]
        else:
            raise TypeError("texts deve ser uma pd.Series, numpy.ndarray ou iterável de strings.")
        return iterable

    @torch.no_grad()
    def _embed_batches(self, texts: Sequence[str]) -> np.ndarray:
        outputs: List[np.ndarray] = []
        total = len(texts)
        start_global = time.perf_counter()
        if self.verbose:
            print(f"[BERT] Iniciando embedding de {total} textos em batches de {self.batch_size} no device '{self.device}'...")

        for i in range(0, total, self.batch_size):
            batch = texts[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            model_out = self.model(**inputs)
            # model_out.last_hidden_state: (batch, seq_len, hidden)
            token_embeddings = model_out.last_hidden_state
            attention_mask = inputs.get("attention_mask")

            # Mean pooling sobre tokens válidos
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

            outputs.append(mean_pooled.cpu().numpy())

            # Progresso
            if self.verbose:
                processed = min(i + self.batch_size, total)
                if (processed // self.batch_size) % self.progress_every == 0 or processed == total:
                    elapsed = time.perf_counter() - start_global
                    rate = processed / elapsed if elapsed > 0 else 0.0
                    remaining = total - processed
                    eta = remaining / rate if rate > 0 else float("inf")
                    pct = (processed / total) * 100 if total else 100.0
                    eta_str = (
                        f"{int(eta // 60):02d}:{int(eta % 60):02d}"
                        if eta != float("inf")
                        else "--:--"
                    )
                    print(
                        f"[BERT] {processed}/{total} ({pct:.1f}%) | {elapsed:.1f}s passados | ETA ~ {eta_str}"
                    )

        return np.vstack(outputs)
